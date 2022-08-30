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


# some tensor ops should not support shard, i.e. local_scalar_dense
# shouldn't work for shard as it requires numel == 1
def no_shard_prop_rule(op_schema: OpSchema) -> OutputSharding:
    # by default prop the first arg spec
    tensor_spec = op_schema.args_spec[0]
    for placement in tensor_spec.placements:
        if placement.is_shard():
            return OutputSharding(
                None,
                failed_reason=f"Op does not support input placements "
                f"with `Shard`, but found placements: "
                f"{tensor_spec.placements}",
            )
    # otherwise default prop the first arg spec
    return OutputSharding(tensor_spec)


default_prop_ops = [
    "aten._to_copy.default",
    "aten.clone.default",
    "aten.copy_.default",
    "aten.detach.default",
    "aten.is_same_size.default",
    "aten.ones_like.default",
    "aten.new_empty_strided.default",
]

no_shard_prop_ops = [
    "aten._local_scalar_dense.default",
]

for op in default_prop_ops:
    DTensor._op_to_rules[op] = default_prop_rule

for op in no_shard_prop_ops:
    DTensor._op_to_rules[op] = no_shard_prop_rule
