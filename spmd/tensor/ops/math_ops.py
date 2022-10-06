# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast

from spmd.tensor.api import DTensor
from spmd.tensor.placement_types import DTensorSpec, Replicate, _Partial, Shard
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.common_rules import reduction_rule, pointwise_rule
from spmd.tensor.ops.utils import register_prop_rule


reduction_ops = [
    "aten.all.default",
    "aten.sum.SymInt",
    "aten.sum.default",
    "aten.sum.dim_IntList",
]

for reduction_op in reduction_ops:
    DTensor._op_to_rules[reduction_op] = reduction_rule


@register_prop_rule("aten._softmax.default")
def softmax_rule(op_schema: OpSchema) -> OutputSharding:
    input_spec, softmax_dim, _ = op_schema.args_schema
    input_spec = cast(DTensorSpec, input_spec)
    softmax_dim = cast(int, softmax_dim)
    dim_map = input_spec.dim_map
    if softmax_dim < len(dim_map) and dim_map[softmax_dim] >= 0:
        raise RuntimeError("Cannot run softmax on sharding dimension!")
    return OutputSharding(input_spec)


@register_prop_rule("aten._softmax_backward_data.default")
def softmax_bwd_rule(op_schema: OpSchema) -> OutputSharding:
    grad_out_spec, out_spec, softmax_dim, _ = op_schema.args_schema
    grad_out_spec = cast(DTensorSpec, grad_out_spec)
    out_spec = cast(DTensorSpec, out_spec)
    softmax_dim = cast(int, softmax_dim)
    grad_out_dim_map = grad_out_spec.dim_map
    out_dim_map = out_spec.dim_map
    if softmax_dim < len(grad_out_dim_map) and (
        grad_out_dim_map[softmax_dim] >= 0 or out_dim_map[softmax_dim] >= 0
    ):
        raise RuntimeError(
            "Cannot run _softmax_backward_data on sharding dimension!"
        )
    return pointwise_rule(op_schema)


@register_prop_rule("aten.native_dropout.default")
def dropout_rule(op_schema: OpSchema) -> OutputSharding:
    self_spec = cast(DTensorSpec, op_schema.args_schema[0])

    # TODO: enable dropout on partial/replicate tensor when we fix
    # the problem of non-deterministic algorithm with replication.
    replicate_or_partial = False
    for placement in self_spec.placements:
        if isinstance(placement, (Replicate, _Partial)):
            replicate_or_partial = True
            break

    if replicate_or_partial:
        return OutputSharding(
            None, failed_reason="Dropout with replication is not supported yet!"
        )
    else:
        return OutputSharding(self_spec)
