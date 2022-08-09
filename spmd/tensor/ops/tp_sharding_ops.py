# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import math

import torch
import torch.utils._pytree as pytree
from typing import List
from spmd.tensor.api import DTensor
from spmd.tensor.placement_types import Shard
from spmd.tensor.utils import unwrap_local_tensor
from spmd.tensor.ops.utils import unwrap_single_placement, register_impl

"""
The ops below were quickly hacked and needed to be polished down the road.
Although they come with unit tests already, the logic is directly borrowed
from ShardedTensor. We need to also make it work for all placement types 
of DTensor and all corner cases for sharded distributed tensor.
"""


@register_impl("aten.view.default")
# pyre-fixme[2]: Parameter must be annotated.
def dist_view(self: DTensor, *shape) -> DTensor:
    mat_placement = pytree.tree_map(unwrap_single_placement, self)
    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    if mat_placement.is_replicate():
        return DTensor.from_local(
            local_mat.view(*shape), self.device_mesh, [mat_placement]
        )

    elif mat_placement.is_shard():
        shape = shape[0]
        try:
            infer_idx = shape.index(-1)
        except ValueError:
            infer_idx = None  # type: ignore

        # Infer the dim which is specified with -1.
        if infer_idx is not None:
            st_size = math.prod(self.size())  # type: ignore[attr-defined]
            shape_size = -1 * math.prod(shape)  # type: ignore[attr-defined]
            # pyre-fixme[60]: Concatenation not yet support for multiple variadic
            shape = (
                *shape[:infer_idx],
                st_size // shape_size,
                *shape[infer_idx + 1 :],
            )
        if self.size() == shape:
            return self

        sharding_dim = mat_placement.dim
        # When the sharding dim is negative, we need to ensure the new
        # sharded tensor is still sharded by the original dimension.
        if sharding_dim < 0:
            sharding_dim = self.dim() + sharding_dim

        world_size = self.device_mesh.size(dim=0)
        if shape[sharding_dim] % world_size:
            raise NotImplementedError(
                f"Case when dim '({shape[sharding_dim]})' is not divisible "
                "by world_size is not supported."
            )
        # pyre-fixme[60]: Concatenation not yet support for multiple variadic
        new_local_tensor_size = (
            *shape[:sharding_dim],
            shape[sharding_dim] // world_size,
            *shape[sharding_dim + 1 :],
        )
        new_local_tensor = local_mat.view(*new_local_tensor_size)
        return DTensor(new_local_tensor, self.device_mesh, self.placements)
    else:
        raise RuntimeError("not supported!")


@register_impl("aten.transpose.int")
def dist_transpose(self: DTensor, dim0: int, dim1: int) -> DTensor:
    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    mat_placement = pytree.tree_map(unwrap_single_placement, self)
    device_mesh = self.device_mesh
    new_shard_dim = (
        dim1 if mat_placement.is_shard(dim=dim0) else mat_placement.dim
    )
    new_shard_dim = dim0 if mat_placement.is_shard(dim=dim1) else new_shard_dim
    new_sharding_placement = [Shard(new_shard_dim)]
    local_tensor = local_mat.transpose(dim0, dim1)
    return DTensor(local_tensor, device_mesh, new_sharding_placement)


@register_impl("aten.baddbmm.default")
def dist_baddbmm(
    self: DTensor,
    batch1: DTensor,
    batch2: DTensor,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> DTensor:
    local_input, local_batch1, local_batch2 = pytree.tree_map(
        unwrap_local_tensor, (self, batch1, batch2)
    )
    local_tensor = torch.ops.aten.baddbmm(
        local_input, local_batch1, local_batch2, beta=beta, alpha=alpha
    )
    return DTensor(local_tensor, self.device_mesh, self.placements)


@register_impl("aten.bmm.default")
def dist_bmm(self: DTensor, mat2: DTensor) -> DTensor:
    local_input, local_mat2 = pytree.tree_map(unwrap_local_tensor, (self, mat2))
    local_tensor = torch.ops.aten.bmm(local_input, local_mat2)
    return DTensor(local_tensor, self.device_mesh, self.placements)


@register_impl("aten._softmax.default")
def dist_softmax(self: DTensor, dim: int, half_to_float: bool) -> DTensor:
    local_input = pytree.tree_map(unwrap_local_tensor, (self))
    local_tensor = local_input.softmax(dim=dim)
    return DTensor(local_tensor, self.device_mesh, self.placements)


@register_impl("aten.permute.default")
def dist_permute(self: DTensor, dims: List[int]) -> DTensor:
    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    mat_placement = pytree.tree_map(unwrap_single_placement, self)

    if mat_placement.is_replicate():
        local_tensor = torch.ops.aten.permute(local_mat, dims=dims)
        return DTensor(
            local_tensor, self.device_mesh, [mat_placement], run_check=False
        )
    elif mat_placement.is_shard():
        sharding_dim = mat_placement.dim
        new_sharding_dim = dims.index(sharding_dim)
        new_sharding_placement = [Shard(new_sharding_dim)]
        local_tensor = torch.ops.aten.permute(local_mat, dims=dims)
        return DTensor(local_tensor, self.device_mesh, new_sharding_placement)
    else:
        raise RuntimeError("Not supported!")


@register_impl("aten.cat.default")
def dist_cat(tensor_list: List[DTensor], dim: int = 0) -> DTensor:
    local_inputs = pytree.tree_map(unwrap_local_tensor, tensor_list)
    local_tensor = torch.ops.aten.concat(local_inputs, dim=dim)
    return DTensor(
        local_tensor, tensor_list[0].device_mesh, tensor_list[0].placements
    )


@register_impl("aten.split.Tensor")
# pyre-fixme[2]: Parameter must be annotated.
def dist_split(self: DTensor, split_size_or_sections, dim=0) -> List[DTensor]:
    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    mat_placement = pytree.tree_map(unwrap_single_placement, self)
    sharding_dim = mat_placement.dim
    world_size = self.device_mesh.size(dim=0)
    if dim < 0:
        dim = self.dim() + dim
    if sharding_dim < 0:
        sharding_dim = self.dim() + sharding_dim
    if dim == sharding_dim:
        if type(split_size_or_sections) is list:
            split_size_or_sections[sharding_dim] //= world_size
        else:
            split_size_or_sections //= world_size
    tensor_list = local_mat.split(split_size_or_sections, dim=dim)
    return [
        DTensor(tensor, self.device_mesh, [mat_placement])
        for tensor in tensor_list
    ]


@register_impl("contiguous")
# pyre-fixme[2]: Parameter must be annotated.
def dist_contiguous(self) -> "DTensor":
    return DTensor(
        self._local_tensor.contiguous(), self.device_mesh, self.placements
    )


@register_impl("is_contiguous")
# pyre-fixme[2]: Parameter must be annotated.
def dist_is_contiguous(self) -> bool:
    return self.to_local().is_contiguous()
