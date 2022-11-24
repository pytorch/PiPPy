# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast, Optional, Sequence

import torch
from spmd.tensor.api import DTensor
from spmd.tensor.placement_types import DTensorSpec
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.common_rules import reduction_rule, pointwise_rule
from spmd.tensor.ops.utils import register_prop_rule, as_list, normalize_dims


def _infer_reduction_dims(
    dims_arg: object, ndim: int
) -> Optional[Sequence[int]]:
    if dims_arg is None:
        return None
    dims = cast(Sequence[int], as_list(dims_arg))
    dims = normalize_dims(dims, ndim)
    empty_dims = [[0], [-1], []]
    if ndim == 0 and dims_arg in empty_dims:
        return None
    return dims


@register_prop_rule("aten.all.default")
def default_reduction_rule(op_schema: OpSchema) -> OutputSharding:
    return reduction_rule(op_schema, reduction_linear=True)


def sum_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_spec.ndim)

    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return reduction_rule(
        op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=True
    )


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


def mean_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    # if length of args > 1, we check args to find dims
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_spec.ndim)

    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return reduction_rule(
        op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=False
    )


mean_ops = [
    "aten.mean.default",
    "aten.mean.dim",
    "aten.mean.out",
]

for mean_op in mean_ops:
    DTensor._op_to_rules[mean_op] = mean_rule


def var_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    # if length of args > 1, we check args to find dims, note that
    # var.default have unbias arg as the first argument, so we want
    # to check if it's not bool
    if len(args_schema) > 1 and not isinstance(args_schema[1], bool):
        dims = _infer_reduction_dims(args_schema[1], input_spec.ndim)

    keep_dim = len(args_schema) > 3 and bool(args_schema[3])
    return reduction_rule(
        op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=False
    )


var_ops = [
    "aten.var.default",
    "aten.var.dim",
    "aten.var.out",
]

for var_op in var_ops:
    DTensor._op_to_rules[var_op] = var_rule


@register_prop_rule("aten.var.correction")
@register_prop_rule("aten.var.correction_out")
def var_correction_rule(op_schema: OpSchema) -> OutputSharding:
    args_schema = op_schema.args_schema
    input_spec = cast(DTensorSpec, args_schema[0])
    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_spec.ndim)

    # keep_dim is a kwarg instead of arg for var.correction
    keep_dim = cast(bool, op_schema.kwargs_schema.get("keepdim", False))
    return reduction_rule(
        op_schema, dims=dims, keep_dim=keep_dim, reduction_linear=False
    )


@register_prop_rule("aten.native_layer_norm.default")
def _prop_native_layer_norm(op_schema: OpSchema) -> OutputSharding:
    # TODO: Assert placement is Replicate() since we need to insert 
    # allreduce calls for _Partial results.
    input, normalized_shape, weight, bias, eps = op_schema.args_schema
    out, mean, rstd = input, weight, weight
    dims_list = op_schema.args_schema[1]
    assert isinstance(input, DTensorSpec)
    return OutputSharding(
        output_spec=(DTensorSpec(mesh=input.mesh,
        placements=tuple(input.placements),
        ndim=input.ndim,
        shape=torch.Size((s for i, s in enumerate(input.shape)))
        ), DTensorSpec(
        mesh=input.mesh,
        placements=tuple(input.placements),
        ndim=input.ndim,
        shape=torch.Size((s for i, s in enumerate(input.shape)))
        ), DTensorSpec(
        mesh=input.mesh,
        placements=tuple(input.placements),
        ndim=input.ndim,
        shape=torch.Size((s for i, s in enumerate(input.shape)))
        )))
