# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Optional
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpInfo
from spmd.tensor.placement_types import PlacementSpec


def default_prop_rule(op_info: OpInfo) -> Optional[PlacementSpec]:
    # by default prop the first arg spec
    return op_info.args_spec[0]

default_prop_ops = [
    "aten.is_same_size.default",
    "aten.ones_like.default",
    "aten.detach.default",
]
for op in default_prop_ops:
    DTensor._op_to_rules[op] = default_prop_rule

# @register_impl("aten.expand.default")
# def dist_expand(types, args=(), kwargs=None):
#     self_tensor = args[0]
#     device_mesh = self_tensor.device_mesh

#     new_local_tensor = torch.ones_like(self_tensor.to_local())
#     return DTensor.from_local(new_local_tensor, device_mesh, self_tensor.placements)
