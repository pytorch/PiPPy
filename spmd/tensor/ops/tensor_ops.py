# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding


# NOTE: the default propagation rule should apply for
# any operator that does not return a DTensor, i.e.
# for operators that only returns int/float/bool, we by
# default still propagate the spec, this is to ensure
# that we only return None for the case where the sharding
# propagation failed, and we should do auto-redistribute
def default_prop_rule(op_schema: OpSchema) -> OutputSharding:
    # by default prop the first arg spec
    return OutputSharding(op_schema.args_spec[0])


default_prop_ops = [
    "aten._local_scalar_dense.default",
    "aten.clone.default",
    "aten.copy_.default",
    "aten.detach.default",
    "aten.is_same_size.default",
    "aten.ones_like.default",
    "aten.new_empty_strided.default",
]
for op in default_prop_ops:
    DTensor._op_to_rules[op] = default_prop_rule
