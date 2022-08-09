# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema
from spmd.tensor.placement_types import PlacementSpec


# NOTE: the default propagation rule should apply for
# any operator that does not return a DTensor, i.e.
# for operators that only returns int/float/bool, we by
# default still propagate the spec, this is to ensure
# that we only return None for the case where the sharding
# propagation failed, and we should do auto-redistribute
def default_prop_rule(op_schema: OpSchema) -> PlacementSpec:
    # by default prop the first arg spec
    return op_schema.args_spec[0]


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
