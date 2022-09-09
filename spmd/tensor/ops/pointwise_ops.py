# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.math_ops import einop_rule
from spmd.tensor.ops.utils import register_prop_rule
from typing import cast
from spmd.tensor.placement_types import DTensorSpec
import torch.utils._pytree as pytree

# leave the pointwise_ops list here for convenience,
# it might not be a complete list.
# TODO: enable the pointwise ops listed below, and test
# all of them properly.
# pointwise_ops = [
#     "abs",
#     "absolute",
#     "acos",
#     "arccos",
#     "acosh",
#     "arccosh",
#     "add",
#     "addcdiv",
#     "addcmul",
#     "angle",
#     "asin",
#     "arcsin",
#     "asinh",
#     "arcsinh",
#     "atan",
#     "arctan",
#     "atanh",
#     "arctanh",
#     "atan2",
#     "arctan2",
#     "bitwise_not",
#     "bitwise_and",
#     "bitwise_or",
#     "bitwise_xor",
#     "bitwise_left_shift",
#     "bitwise_right_shift",
#     "ceil",
#     "clamp",
#     "clip",
#     "conj_physical",
#     "copysign",
#     "cos",
#     "cosh",
#     "deg2rad",
#     "div",
#     "divide",
#     "digamma",
#     "erf",
#     "erfc",
#     "erfinv",
#     "exp",
#     "exp2",
#     "expm1",
#     "fake_quantize_per_channel_affine",
#     "fake_quantize_per_tensor_affine",
#     "fix",
#     "float_power",
#     "floor",
#     "floor_divide",
#     "fmod",
#     "frac",
#     "frexp",
#     "gradient",
#     "imag",
#     "ldexp",
#     "lerp",
#     "lgamma",
#     "log",
#     "log10",
#     "log1p",
#     "log2",
#     "logaddexp",
#     "logaddexp2",
#     "logical_and",
#     "logical_not",
#     "logical_or",
#     "logical_xor",
#     "logit",
#     "hypot",
#     "i0",
#     "igamma",
#     "igammac",
#     "mul",
#     "multiply",
#     "mvlgamma",
#     "nan_to_num",
#     "neg",
#     "negative",
#     "nextafter",
#     "polygamma",
#     "positive",
#     "pow",
#     "quantized_batch_norm",
#     "quantized_max_pool1d",
#     "quantized_max_pool2d",
#     "rad2deg",
#     "real",
#     "reciprocal",
#     "remainder",
#     "round",
#     "rsqrt",
#     "sigmoid",
#     "sign",
#     "sgn",
#     "signbit",
#     "sin",
#     "sinc",
#     "sinh",
#     "sqrt",
#     "square",
#     "sub",
#     "subtract",
#     "tan",
#     "tanh",
#     "true_divide",
#     "trunc",
#     "xlogy",
# ]

pointwise_ops = [
    "aten.add.Tensor",
    "aten.add.Scalar",
    "aten.add_.Tensor",
    "aten.add_.Scalar",
    "aten.add.out",
    "aten.abs.default",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_and_.Tensor",
    "aten.bitwise_or_.Tensor",
    "aten.eq.Tensor",
    "aten.gelu.default",
    "aten.relu.default",
    "aten.le.Tensor",
    "aten.sigmoid.default",
    "aten.sub.Tensor",
    "aten.threshold_backward.default",
    "aten.isnan.default",
    "aten.mul.Scalar",
    "aten.mul_.Scalar",
    "aten.mul.Tensor",
    "aten.mul.out",
    "aten.mul_.Tensor",
    "aten.ne.Scalar",
    "aten.tanh.default",
    "aten.tanh_backward.default",
    "aten.sgn.default",
    "aten.neg.default",
]


def pointwise_rule(
    op_schema: OpSchema, linearity: bool = False
) -> OutputSharding:
    """
    Propagate the sharding for pointwise operations. Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # handle the case of broadcasting, find the max_dim first
    # TODO: handle the "broadcasting to a common shape case"
    # TODO: handle inplace op properly without run propagation
    input_specs = op_schema.args_spec
    max_dim = max(input.ndim for input in input_specs)
    dimchars = []
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"
    return einop_rule(fmt, op_schema, linearity=linearity)


for op in pointwise_ops:
    DTensor._op_to_rules[op] = pointwise_rule

@register_prop_rule("aten._softmax_backward_data.default")
def softmax_bwd_rule(op_schema: OpSchema) -> OutputSharding:
    if (len(op_schema.args_schema) != len(op_schema.args_spec)):
        ops_dim_map = pytree.tree_map(
            lambda spec: cast(DTensorSpec, spec).dim_map, op_schema.args_spec
        )
        softmax_dim = cast(
            int, op_schema.args_schema[len(op_schema.args_spec)]
        )
        ops_dim_map = list(zip(*ops_dim_map))
        if softmax_dim < len(ops_dim_map) and 0 in ops_dim_map[softmax_dim]:
            raise RuntimeError("Cannot run softmax on batch dim!")
    return pointwise_rule(op_schema)
