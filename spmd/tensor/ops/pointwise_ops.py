# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict
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


def _replace_char_in_str(string: str, new_char: str, idx: int) -> str:
    return string[:idx] + new_char + string[idx + 1 :]


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
    dimchar_singleton_counter: Dict[str, int] = {}
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        # handle the "broadcasting to a common shape case"
        # see https://pytorch.org/docs/stable/notes/broadcasting.html
        # If any of the dimensions is singleton dimension (i.e. 1).
        # we mark the dim char as a special "1" to distinguish with
        # the non-singleton dimension, so that sharding propagation
        # should just ignore the singleton dimension.
        if len(input_specs) > 1:
            for i, dim_length in enumerate(input.shape):
                if dim_length == 1:
                    # mark singleton dim char as a special "1" in einop rule
                    dimchar_singleton_counter[p[i]] = (
                        dimchar_singleton_counter.get(p[i], 0) + 1
                    )
                    p = _replace_char_in_str(p, "1", i)
        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    # check if we replace the all inputs dim char with singleton dimension,
    # if we replace all inputs, we also need to replace the output dimension.
    for output_dim_idx in range(len(out_dimchars)):
        out_dimchar = out_dimchars[output_dim_idx]
        if dimchar_singleton_counter.get(out_dimchar, 0) == len(input_specs):
            out_dimchars = _replace_char_in_str(
                out_dimchars, "1", output_dim_idx
            )

    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"
    return einop_rule(fmt, op_schema, linearity=linearity)


for op in pointwise_ops:
    DTensor._op_to_rules[op] = pointwise_rule
