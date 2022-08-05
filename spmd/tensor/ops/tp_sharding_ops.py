# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import math

import torch
import torch.utils._pytree as pytree
from typing import List
from spmd.tensor.api import Tensor
from spmd.tensor.placement_types import Shard
from spmd.tensor.ops.utils import (
    unwrap_local_tensor,
    unwrap_single_placement,
    register_impl,
)

"""
The ops below were quickly hacked and needed to be polished down the road.
Although they come with unit tests already, the logic is directly borrowed
from ShardedTensor. We need to also make it work for all placement types 
of Distributed Tensor and all corner cases for sharded distributed tensor.
"""


@register_impl("aten.view.default")
def dist_view(self: Tensor, *shape) -> Tensor:
    shape = shape[0]
    try:
        infer_idx = shape.index(-1)
    except ValueError:
        infer_idx = None  # type: ignore

    # Infer the dim which is specified with -1.
    if infer_idx is not None:
        st_size = math.prod(self.size())  # type: ignore[attr-defined]
        shape_size = -1 * math.prod(shape)  # type: ignore[attr-defined]
        shape = (
            *shape[:infer_idx],
            st_size // shape_size,
            *shape[infer_idx + 1 :],
        )
    if self.size() == shape:
        return self

    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    mat_placement = pytree.tree_map(unwrap_single_placement, self)

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
    new_local_tensor_size = (
        *shape[:sharding_dim],
        shape[sharding_dim] // world_size,
        *shape[sharding_dim + 1 :],
    )
    new_local_tensor = local_mat.view(*new_local_tensor_size)
    return Tensor.from_local(
        new_local_tensor, self.device_mesh, self.placements
    )


@register_impl("aten.transpose.int")
def dist_transpose(self: Tensor, dim0: int, dim1: int) -> Tensor:
    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    mat_placement = pytree.tree_map(unwrap_single_placement, self)
    device_mesh = self.device_mesh
    new_shard_dim = (
        dim1 if mat_placement.is_shard(dim=dim0) else mat_placement.dim
    )
    new_shard_dim = dim0 if mat_placement.is_shard(dim=dim1) else new_shard_dim
    new_sharding_placement = [Shard(new_shard_dim)]
    local_tensor = local_mat.transpose(dim0, dim1)
    return Tensor.from_local(local_tensor, device_mesh, new_sharding_placement)


@register_impl("aten.baddbmm.default")
def dist_baddbmm(
    self: Tensor, batch1: Tensor, batch2: Tensor, beta=1.0, alpha=1.0
) -> Tensor:
    local_input, local_batch1, local_batch2 = pytree.tree_map(
        unwrap_local_tensor, (self, batch1, batch2)
    )
    local_tensor = torch.ops.aten.baddbmm(
        local_input, local_batch1, local_batch2, beta=beta, alpha=alpha
    )
    return Tensor.from_local(local_tensor, self.device_mesh, self.placements)


@register_impl("aten.bmm.default")
def dist_bmm(self: Tensor, mat2: Tensor) -> Tensor:
    local_input, local_mat2 = pytree.tree_map(unwrap_local_tensor, (self, mat2))
    local_tensor = torch.ops.aten.bmm(local_input, local_mat2)
    return Tensor.from_local(local_tensor, self.device_mesh, self.placements)


@register_impl("aten._softmax.default")
def dist_softmax(self: Tensor, dim: int, half_to_float: bool) -> Tensor:
    local_input = pytree.tree_map(unwrap_local_tensor, (self))
    local_tensor = local_input.softmax(dim=dim)
    return Tensor.from_local(local_tensor, self.device_mesh, self.placements)


@register_impl("aten.permute.default")
def dist_permute(self: Tensor, dims: List[int]) -> Tensor:
    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    mat_placement = pytree.tree_map(unwrap_single_placement, self)

    sharding_dim = mat_placement.dim
    new_sharding_dim = dims.index(sharding_dim)
    new_sharding_placement = [Shard(new_sharding_dim)]
    local_tensor = torch.ops.aten.permute(local_mat, dims=dims)
    return Tensor.from_local(
        local_tensor, self.device_mesh, new_sharding_placement
    )


@register_impl("aten.cat.default")
def dist_cat(tensor_list: List[Tensor], dim: int = 0) -> Tensor:
    local_inputs = pytree.tree_map(unwrap_local_tensor, tensor_list)
    local_tensor = torch.ops.aten.concat(local_inputs, dim=dim)
    return Tensor.from_local(
        local_tensor, tensor_list[0].device_mesh, tensor_list[0].placements
    )


@register_impl("aten.split.Tensor")
def dist_split(self: Tensor, split_size_or_sections, dim=0) -> List[Tensor]:
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
        Tensor.from_local(tensor, self.device_mesh, [mat_placement])
        for tensor in tensor_list
    ]


@register_impl("contiguous")
def dist_contiguous(self) -> "Tensor":
    return Tensor.from_local(
        self._local_tensor.contiguous(), self.device_mesh, self.placements
    )


@register_impl("is_contiguous")
def dist_is_contiguous(self) -> bool:
    return self.local_tensor().is_contiguous()
