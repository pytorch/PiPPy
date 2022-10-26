import warnings
import functools
import copy
from typing import List, NamedTuple, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor

import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import (
    ChunkShardingSpec,
)

from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)

from torch.distributed._shard.sharding_spec import (
    ShardMetadata,
)

from torch.distributed.remote_device import _remote_device

from spmd.tensor import (
    DTensor as DistributedTensor,
    DeviceMesh,
    Shard as DShard,
)
from spmd.tensor.placement_types import Placement

__all__ = ["is_available"]


class _STShardingInfo(NamedTuple):
    """:class:`ShardedTensor` sharding information."""

    sharding_spec: Optional[shard_spec.ShardingSpec]
    global_size: Optional[torch.Size]
    process_group: Optional[c10d.ProcessGroup]
    device_mesh: Optional[DeviceMesh]
    placements: Optional[List[Placement]]


def _get_box(tensor: DistributedTensor) -> Tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    assert device_mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

    placement = tensor.placements[0]
    offsets = [0] * len(tensor.size())
    num_chunks = device_mesh.size(dim=0)

    if tensor.placements[0].is_shard():
        shard_dim = cast(DShard, placement).dim
        chunk_size = tensor.size(shard_dim) // num_chunks
        offsets[shard_dim] = chunk_size

    return (torch.Size(offsets), tensor._local_tensor.size())


def _get_box_for(
    tensor: DistributedTensor, idx: int
) -> Tuple[torch.Size, torch.Size]:
    offsets, size = _get_box(tensor)
    return (torch.Size([val * idx for val in offsets]), size)


def _get_local_box(tensor: DistributedTensor) -> Tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    dim_0_coord = device_mesh.get_coordinate_on_dim(0)
    assert dim_0_coord is not None
    return _get_box_for(tensor, dim_0_coord)


def _create_shard_md_from_dt(
    dt: DistributedTensor, current_rank: int
) -> ShardMetadata:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

    offsets, sizes = _get_local_box(dt)
    return ShardMetadata(
        shard_offsets=list(offsets),
        shard_sizes=list(sizes),
        placement=f"rank:{current_rank}/{dt._local_tensor.device}",
    )


def _create_sharded_tensor_md_from_dt(
    dt: DistributedTensor, dt_pg: c10d.ProcessGroup
) -> ShardedTensorMetadata:
    # This is where it gets tricky, we have to produce a ShardedTensor that has full coverage
    # and yet has only one valid shard for the current rank.

    shards_md = []
    my_rank = dist.get_rank(dt_pg)
    scapegoat_rank = 0 if my_rank > 0 else 1

    if dt.placements[0].is_shard():
        shard_count = dt_pg.size()
    else:
        shard_count = 1

    for i in range(shard_count):
        offsets, sizes = _get_box_for(dt, i)
        shards_md.append(
            ShardMetadata(
                shard_offsets=list(offsets),
                shard_sizes=list(sizes),
                placement=f"rank:{scapegoat_rank if i > 0 else my_rank}/{dt._local_tensor.device}",
            )
        )

    return ShardedTensorMetadata(
        shards_metadata=shards_md,
        size=dt.size(),
        tensor_properties=TensorProperties(
            dtype=dt.dtype,
            layout=dt.layout,
            requires_grad=dt.requires_grad,
            # ignore memory_format and pin_memory as those are not supported by DT
        ),
    )


def _get_dt_pg(dt: DistributedTensor) -> c10d.ProcessGroup:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"
    return mesh.get_dim_groups()[0]


def _rewrite_spec_if_needed(
    spec: shard_spec.ShardingSpec, tensor: torch.Tensor, rank: int
) -> shard_spec.ShardingSpec:
    """
    Rewrite ``spec`` to match the device of ``tensor``.

    FSDP.sharded_optim_state_dict sneakly ships optimizer state to CPU so if the original ShardingSpec
    produces CUDA metadata, ST construction bombs.
    """
    if not isinstance(spec, ChunkShardingSpec):
        return spec

    # let's see if we need
    rewrite = False
    for p in spec.placements:
        p = cast(_remote_device, p)
        if p.rank() == rank and p.device() != tensor.device:
            rewrite = True
            break
    if rewrite:
        spec = copy.deepcopy(spec)
        for i, placement in enumerate(spec.placements):
            placement = cast(_remote_device, placement)
            if placement.rank() == rank and placement.device() != tensor.device:
                spec.placements[i] = _remote_device(
                    f"rank:{rank}/{tensor.device}"
                )

    return spec


def _flatten_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[_STShardingInfo]]:
    if type(tensor) is ShardedTensor:
        return tensor.local_tensor(), _STShardingInfo(
            tensor.sharding_spec(),
            tensor.size(),
            tensor._process_group,
            None,
            None,
        )
    elif type(tensor) is DistributedTensor:
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, _STShardingInfo(
            None,
            None,
            None,
            tensor.device_mesh,
            list(tensor.placements),
        )
    return tensor, None


def _unflatten_tensor(
    tensor: torch.Tensor, sharding_info: _STShardingInfo
) -> torch.Tensor:
    result: torch.Tensor

    if sharding_info.sharding_spec is not None:
        assert sharding_info.global_size is not None
        result = ShardedTensor._init_from_local_tensor(
            tensor,
            _rewrite_spec_if_needed(
                sharding_info.sharding_spec,
                tensor,
                dist.get_rank(sharding_info.process_group),
            ),
            sharding_info.global_size,
            process_group=cast(dist.ProcessGroup, sharding_info.process_group),
        )
    else:

        def _dt_gradient_hook(
            param: DistributedTensor, grad: DistributedTensor
        ) -> None:
            param.grad = grad
            param._local_tensor.grad = grad._local_tensor

        result = dtensor = DistributedTensor.from_local(
            tensor,
            device_mesh=sharding_info.device_mesh,
            placements=sharding_info.placements,
            run_check=False,
        )
        if dtensor.requires_grad:
            dtensor.register_hook(functools.partial(_dt_gradient_hook, dtensor))

    _set_fsdp_flattened(result)
    return result


def _chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
) -> torch.Tensor:
    if type(tensor) is ShardedTensor:
        assert len(tensor.local_shards()) == 1

        inner_param = tensor.local_tensor()
        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )

        outer_local_shard = tensor.local_shards()[0]
        shards: List[Shard] = [
            Shard(inner_st, copy.deepcopy(outer_local_shard.metadata))
        ]
        st_meta = copy.deepcopy(tensor.metadata())
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=tensor._process_group,
            init_rrefs=False,
        )
        return st_outer
    elif type(tensor) is DistributedTensor:
        device_mesh = tensor.device_mesh
        assert device_mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

        inner_param = tensor._local_tensor

        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            torch.cuda.device_count(),
            pg,
        )

        dt_pg = _get_dt_pg(tensor)
        # We do this differently here, we create a ST with no local shards then patch it
        shards = [
            Shard(
                inner_st, _create_shard_md_from_dt(tensor, dist.get_rank(dt_pg))
            )
        ]

        st_meta = _create_sharded_tensor_md_from_dt(tensor, dt_pg)
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=dt_pg,
            init_rrefs=False,
        )

        return st_outer
    else:
        return _create_chunk_sharded_tensor(
            tensor,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )


def _pre_load_state_dict(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, List[Shard]]:
    shards = cast(ShardedTensor, tensor).local_shards()
    if len(shards) == 1 and type(shards[0].tensor) is ShardedTensor:
        inner_tensor = cast(ShardedTensor, shards[0].tensor)
        shards = inner_tensor.local_shards()
        tensor = inner_tensor

    return (tensor, shards if len(shards) > 0 else [])


try:
    from torch.distributed.fsdp._fsdp_extensions import (
        _set_fsdp_extensions,
        FSDPExtensions,
    )
    from torch.distributed.fsdp._utils import _set_fsdp_flattened

    class DTensorExtensions(FSDPExtensions):
        def pre_flatten_transform(
            self,
            tensor: torch.Tensor,
        ) -> Tuple[torch.Tensor, Optional[_STShardingInfo]]:
            return _flatten_tensor(tensor)

        def post_unflatten_transform(
            self, tensor: torch.Tensor, param_extension: _STShardingInfo
        ) -> torch.Tensor:
            return _unflatten_tensor(tensor, param_extension)

        def chunk_tensor(
            self,
            tensor: torch.Tensor,
            rank: int,
            world_size: int,
            num_devices_per_node: int,
            pg: dist.ProcessGroup,
        ) -> torch.Tensor:
            return _chunk_tensor(
                tensor, rank, world_size, num_devices_per_node, pg
            )

        def pre_load_state_dict_transform(
            self,
            tensor: torch.Tensor,
        ) -> Tuple[torch.Tensor, List[Shard]]:
            return _pre_load_state_dict(tensor)  # type: ignore

    _set_fsdp_extensions(DTensorExtensions())

    def is_available() -> bool:
        return True

except BaseException as e:
    warnings.warn(
        "PyTorch doesn't have TensorFlattener extension point available"
        "2D parallelism won't work with FSDP"
        f"exception: {e}"
    )

    def is_available() -> bool:
        return False
