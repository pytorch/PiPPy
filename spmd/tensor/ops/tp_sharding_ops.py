# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import torch
import torch.utils._pytree as pytree
from typing import List
from spmd.tensor.api import DTensor, Shard
from spmd.tensor.utils import unwrap_local_tensor
from spmd.tensor.ops.utils import unwrap_single_placement, register_impl

"""
The ops below were quickly hacked and needed to be polished down the road.
Although they come with unit tests already, the logic is directly borrowed
from ShardedTensor. We need to also make it work for all placement types
of DTensor and all corner cases for sharded distributed tensor.
"""


@register_impl("aten._unsafe_view.default")
# pyre-fixme[2]: Parameter must be annotated.
def dist_unsafe_view(self: DTensor, *shape) -> DTensor:
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

    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    mat_placement = pytree.tree_map(unwrap_single_placement, self)
    if mat_placement.is_replicate():
        return DTensor(
            local_mat.view(*shape), self.device_mesh, [mat_placement]
        )
    elif mat_placement.is_shard():
        sharding_dim = mat_placement.dim
        placements = self.placements
        if len(shape) <= sharding_dim:
            sharding_dim = 0
            placements = [Shard(sharding_dim)]

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
        new_local_tensor = torch.ops.aten._unsafe_view(
                local_mat, new_local_tensor_size
            )
        return DTensor(
            new_local_tensor,
            self.device_mesh,
            placements,
            requires_grad=new_local_tensor.requires_grad,
        )
    else:
        raise RuntimeError("not supported!")


@register_impl("aten.cat.default")
def dist_cat(tensor_list: List[DTensor], dim: int = 0) -> DTensor:
    local_inputs = pytree.tree_map(unwrap_local_tensor, tensor_list)
    local_tensor = torch.ops.aten.concat(local_inputs, dim=dim)
    return DTensor(
        local_tensor,
        tensor_list[0].device_mesh,
        tensor_list[0].placements,
        requires_grad=local_tensor.requires_grad,
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
        DTensor(
            tensor,
            self.device_mesh,
            [mat_placement],
            requires_grad=tensor.requires_grad,
        )
        for tensor in tensor_list
    ]
