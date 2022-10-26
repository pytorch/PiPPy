# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast, List, Sequence

from spmd.tensor.api import DTensor
from spmd.tensor.placement_types import DTensorSpec
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.common_rules import reduction_rule, pointwise_rule
from spmd.tensor.ops.utils import register_prop_rule, as_list, normalize_dims


@register_prop_rule("aten.all.default")
def all_rule(op_schema: OpSchema) -> OutputSharding:
    return reduction_rule(op_schema, reduction_linear=True)

def sum_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    if len(args_schema) > 1:
        dims = cast(Sequence[int], as_list(args_schema[1]))
        dims = normalize_dims(dims, input_spec.ndim)
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return reduction_rule(op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=True)

sum_ops = [
    "aten.sum.default",
    "aten.sum.dim_IntList",
]
for sum_op in sum_ops:
    DTensor._op_to_rules[sum_op] = sum_rule


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


@register_prop_rule("aten.var.dim")
@register_prop_rule("aten.var.out")
def var_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    # if length of args > 1, we check args to find dims, note that
    # var.default have unbias arg as the first argument, so we want
    # to check if it's not bool
    if len(args_schema) > 1 and not isinstance(args_schema[1], bool):
        dims = cast(Sequence[int], as_list(args_schema[1]))
        dims = normalize_dims(dims, input_spec.ndim)
    keep_dim = len(args_schema) > 3 and bool(args_schema[3])
    return reduction_rule(op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=False)

@register_prop_rule("aten.var.correction")
def var_correction_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    if len(args_schema) > 1 and args_schema[1] is not None:
        dims = cast(Sequence[int], as_list(args_schema[1]))
        dims = normalize_dims(dims, input_spec.ndim)

    # keep_dim is a kwarg instead of arg for var.correction
    keep_dim = op_schema.kwargs_schema.get("keepdim", False)
    return reduction_rule(op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=False)
