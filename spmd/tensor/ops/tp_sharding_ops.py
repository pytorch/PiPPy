# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import torch
import torch.utils._pytree as pytree
from typing import List
from spmd.tensor.api import DTensor
from spmd.tensor.utils import unwrap_local_tensor
from spmd.tensor.ops.utils import unwrap_single_placement, register_impl

"""
The ops below were quickly hacked and needed to be polished down the road.
Although they come with unit tests already, the logic is directly borrowed
from ShardedTensor. We need to also make it work for all placement types
of DTensor and all corner cases for sharded distributed tensor.
"""


@register_impl("aten._softmax.default")
def dist_softmax(self: DTensor, dim: int, half_to_float: bool) -> DTensor:
    local_input = pytree.tree_map(unwrap_local_tensor, (self))
    local_tensor = local_input.softmax(dim=dim)
    return DTensor(local_tensor, self.device_mesh, self.placements)


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
