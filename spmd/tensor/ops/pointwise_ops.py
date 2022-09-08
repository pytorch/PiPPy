# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.math_ops import einop_rule

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
    # find the max_dim first in case we need to broadcasting
    input_specs = op_schema.args_spec
    max_dim = max(input.ndim for input in input_specs)
    dimchars = []
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        # handle the "broadcasting to a common shape case"
        # see https://pytorch.org/docs/stable/notes/broadcasting.html
        # If any of the dimensions is singleton dimension (i.e. 1).
        # we mark the dim char as a special "1" to distinguish with
        # the non-singleton dimension, so that sharding propagation
        # should just ignore the singleton dimension.
        for i, dim_length in enumerate(input.shape):
            if dim_length == 1:
                # mark singleton dim char as a special "1" in einop rule
                p = p[:i] + "1" + p[i + 1 :]
        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"
    return einop_rule(fmt, op_schema, linearity=linearity)


for op in pointwise_ops:
    DTensor._op_to_rules[op] = pointwise_rule
