# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, cast

from spmd.tensor.api import DTensor
from spmd.tensor.placement_types import DTensorSpec, OutputSpecType, Replicate
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.common_rules import (
    reduction_rule,
    pointwise_rule,
    _inplace_rewrap_schema_suggestion,
)
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

    output_spec: OutputSpecType = input_spec
    schema_suggestion = None
    failed_reason = None
    if softmax_dim < len(dim_map) and dim_map[softmax_dim] >= 0:
        # suggest replicating the input tensor
        output_spec = None
        # taken from _inplace_rewrap_schema_suggestion
        new_arg_schema: List[object] = [
            DTensorSpec(
                input_spec.mesh,
                [Replicate()],
                input_spec.shape,
                ndim=input_spec.ndim,
            )
        ]
        new_arg_schema += tuple(op_schema.args_schema[1:])
        schema_suggestion = [
            OpSchema(tuple(new_arg_schema), op_schema.kwargs_schema)
        ]
        failed_reason = "Cannot run softmax on sharding dimension, need to replicate the tensor."
    return OutputSharding(output_spec, schema_suggestion, failed_reason)


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
        schema_suggestion = OpSchema(
            (
                DTensorSpec(
                    grad_out_spec.mesh,
                    [Replicate()],
                    grad_out_spec.shape,
                    ndim=grad_out_spec.ndim,
                ),
                DTensorSpec(
                    out_spec.mesh,
                    [Replicate()],
                    out_spec.shape,
                    ndim=out_spec.ndim,
                ),
            ),
            {},
        )
        failed_reason = "Cannot run _softmax_backward_data on sharding dimension, need to replicate the tensor."
        _inplace_rewrap_schema_suggestion(schema_suggestion, op_schema)
        return OutputSharding(None, [schema_suggestion], failed_reason)
    else:
        return pointwise_rule(op_schema)
