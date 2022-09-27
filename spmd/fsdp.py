import warnings
import functools
import copy
from typing import Any, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

from torch.distributed.fsdp._utils import _set_fsdp_flattened
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor

import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec

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

from spmd.tensor import DTensor as DistributedTensor, DeviceMesh
from spmd.tensor.placement_types import Placement


class STShardingInfo(NamedTuple):
    """:class:`ShardedTensor` sharding information."""

    sharding_spec: Optional[shard_spec.ShardingSpec]
    global_size: Optional[torch.Size]
    process_group: Optional[dist.ProcessGroup]
    device_mesh: Optional[DeviceMesh]
    placements: Optional[List[Placement]]

def get_box(tensor: DistributedTensor) -> Tuple:
    device_mesh = tensor.device_mesh
    assert device_mesh.ndim == 1, f"Only 1D DeviceMeshes currently handled"
    # assert tensor.placements[0].is_shard(), f"Only sharded placement supported"

    placement = tensor.placements[0]
    offsets = [0] * len(tensor.size())
    num_chunks = device_mesh.size(dim=0)

    if tensor.placements[0].is_shard():
        shard_dim = placement.dim
        chunk_size = tensor.size(shard_dim) // num_chunks
        offsets[shard_dim] = chunk_size

    return (torch.Size(offsets), tensor._local_tensor.size())

def get_box_for(tensor: DistributedTensor, idx: int) -> Tuple:
    offsets, size = get_box(tensor)
    offsets = [val * idx for val in offsets]
    return (torch.Size(offsets), size)

def get_local_box(tensor: DistributedTensor) -> Tuple:
    device_mesh = tensor.device_mesh
    return get_box_for(tensor, device_mesh.get_coordinate_on_dim(0))

def create_shard_md_from_dt(dt: DistributedTensor, current_rank) -> ShardMetadata:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, f"Only 1D DeviceMeshes currently handled"

    offsets, sizes = get_local_box(dt)
    return ShardMetadata(
        shard_offsets=list(offsets),
        shard_sizes=list(sizes),
        placement=f"rank:{current_rank}/{dt._local_tensor.device}"
    )

def create_sharded_tensor_md_from_dt(dt: DistributedTensor, dt_pg) -> ShardedTensorMetadata:
    # This is where it goes hilarious, we have to produce a ShardedTensor that has full coverage
    # and yet has only one valid shard for the current rank.

    shards_md = []
    my_rank = dist.get_rank(dt_pg)
    scapegoat_rank = 0 if my_rank > 0 else 1

    if dt.placements[0].is_shard():
        shard_count = dt_pg.size()
    else:
        shard_count = 1 #

    for i in range(shard_count):
        offsets, sizes = get_box_for(dt, i)
        shards_md.append(ShardMetadata(
            shard_offsets=list(offsets),
            shard_sizes=list(sizes),
            placement=f"rank:{scapegoat_rank}/{dt._local_tensor.device}"
        ))

    return ShardedTensorMetadata(
        shards_metadata=shards_md,
        size=dt.size(),
        tensor_properties=TensorProperties(
            dtype=dt.dtype,
            layout=dt.layout,
            requires_grad=dt.requires_grad,
            #ignore memory_format and pin_memory as those are not supported by DT
        )
    )

def get_dt_pg(dt: DistributedTensor) -> dist.ProcessGroup:
    mesh = dt.device_mesh
    assert mesh.ndim == 1, f"Only 1D DeviceMeshes currently handled"
    return mesh.get_dim_groups()[0]


def _rewrite_spec_if_needed(spec: shard_spec.ShardingSpec, tensor: torch.Tensor, rank: int):
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
        if p.rank() == rank and p.device() != tensor.device:
            rewrite = True
            break
    if rewrite:
        spec = copy.deepcopy(spec)
        for i, placement in enumerate(spec.placements):
            if placement.rank() == rank and placement.device() != tensor.device:
                spec.placements[i] = _remote_device(f"rank:{rank}/{tensor.device}")

    return spec

def _flatten_tensor(tensor: torch.Tensor,) -> Tuple[torch.Tensor, Optional[Any]]:
    if type(tensor) is ShardedTensor:
        return tensor.local_tensor(), STShardingInfo(
            tensor.sharding_spec(),
            tensor.size(),
            tensor._process_group,
            None,
            None,
        )
    elif type(tensor) is DistributedTensor:
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, STShardingInfo(
            None,
            None,
            None,
            tensor.device_mesh,
            tensor.placements,
        )
    return tensor, None

def _unflatten_tensor(tensor: torch.Tensor, sharding_info: STShardingInfo) -> torch.Tensor:
    if sharding_info.sharding_spec is not None:
        sharded_tensor = ShardedTensor._init_from_local_tensor(
            tensor,
            _rewrite_spec_if_needed(
                sharding_info.sharding_spec,
                tensor,
                dist.get_rank(sharding_info.process_group)
            ),
            sharding_info.global_size,
            process_group=sharding_info.process_group,
        )
    else:
        def _dt_gradient_hook(param, grad):
                param.grad = grad
                param._local_tensor.grad = grad._local_tensor

        sharded_tensor = DistributedTensor.from_local(
            tensor,
            device_mesh=sharding_info.device_mesh,
            placements=sharding_info.placements,
            run_check=False,
        )
        if sharded_tensor.requires_grad:
            sharded_tensor.register_hook(functools.partial(_dt_gradient_hook, sharded_tensor))

    _set_fsdp_flattened(sharded_tensor)
    return sharded_tensor

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
            init_rrefs=False
        )
        return st_outer
    elif type(tensor) is DistributedTensor:
        device_mesh = tensor.device_mesh
        assert device_mesh.ndim == 1, f"Only 1D DeviceMeshes currently handled"

        inner_param = tensor._local_tensor

        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            torch.cuda.device_count(),
            pg,
        )

        dt_pg = get_dt_pg(tensor)
        # We do this differently here, we create a ST with no local shards then patch it
        shards = []
        st_meta = create_sharded_tensor_md_from_dt(tensor, dt_pg)
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=dt_pg,
            init_rrefs=False
        )
        st_outer._local_shards.append(
            Shard(
                inner_st, 
                create_shard_md_from_dt(tensor, dist.get_rank(dt_pg))
            )
        )

        return st_outer
    else:
        return _create_chunk_sharded_tensor(
            tensor,
            rank,
            world_size,
            num_devices_per_node,
            pg
        )

def _pre_load_state_dict(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    shards = tensor.local_shards()
    if len(shards) == 1 and type(shards[0].tensor) is ShardedTensor:
        tensor = shards[0].tensor
        shards = tensor.local_shards()

    return (tensor, [shards[0].tensor] if len(shards) > 0 else [])



try:
    from torch.distributed.fsdp._tensor_flattener import (
        _set_tensor_flattener,
        TensorFlattener,
    )
    class TensorFlattener2DParallel(TensorFlattener):
        def pre_flatten_transform(
            self,
            tensor: torch.Tensor,
        ) -> Tuple[torch.Tensor, Optional[Any]]:
            return _flatten_tensor(tensor)

        def post_unflatten_transform(
            self, tensor: torch.Tensor, sharding_info: STShardingInfo
        ) -> torch.Tensor:
            return _unflatten_tensor(tensor, sharding_info)

        def chunk_tensor(
            self,
            tensor: torch.Tensor,
            rank: int,
            world_size: int,
            num_devices_per_node: int,
            pg: dist.ProcessGroup,
        ) -> torch.Tensor:
            return _chunk_tensor(
                tensor,
                rank,
                world_size,
                num_devices_per_node,
                pg
            )

        def pre_load_state_dict_transform(
            self,
            tensor: torch.Tensor,
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            return _pre_load_state_dict(tensor)

    _set_tensor_flattener(TensorFlattener2DParallel())
except BaseException as e:
    warnings.warn(
        "PyTorch doesn't have TensorFlattener extension point available"
        "2D parallelism won't work with FSDP"
        f"exception: {e}"
    )
