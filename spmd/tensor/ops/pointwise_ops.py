# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, List
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.math_ops import einop_rule
from spmd.tensor.ops.utils import register_prop_rule
from typing import cast
from spmd.tensor.placement_types import DTensorSpec, Replicate
import torch.utils._pytree as pytree

# leave the remaining pointwise_ops list here for convenience,
# Below ops are some pointwise ops that are yet to be supported,
# they might not be a complete list.
# pointwise_ops = [
#     "fake_quantize_per_channel_affine",
#     "fake_quantize_per_tensor_affine",
#     "floor_divide",  # floor_divide is deprecated
#     "frexp",  # multiple output pointwise op, need to add support
#     "gradient",  #  need investigation on this op
#     "imag",  # complex data type only
#     "quantized_batch_norm",
#     "quantized_max_pool1d",
#     "quantized_max_pool2d",
#     "real",  # complex data type only
# ]

pointwise_ops = [
    "aten.abs.default",
    "aten.acos.default",
    "aten.acos_.default",
    "aten.acos.out",
    "aten.acosh.default",
    "aten.acosh_.default",
    "aten.acosh.out",
    "aten.add.Tensor",
    "aten.add.Scalar",
    "aten.add_.Tensor",
    "aten.add_.Scalar",
    "aten.add.out",
    "aten.addcdiv.default",
    "aten.addcdiv_.default",
    "aten.addcdiv.out",
    "aten.addcmul.default",
    "aten.addcmul_.default",
    "aten.addcmul.out",
    "aten.angle.default",
    "aten.angle.out",
    "aten.asin.default",
    "aten.asin_.default",
    "aten.asin.out",
    "aten.asinh.default",
    "aten.asinh_.default",
    "aten.asinh.out",
    "aten.atan.default",
    "aten.atan_.default",
    "aten.atan.out",
    "aten.atanh.default",
    "aten.atanh_.default",
    "aten.atanh.out",
    "aten.atan2.default",
    "aten.atan2_.default",
    "aten.atan2.out",
    "aten.bitwise_not.default",
    "aten.bitwise_not_.default",
    "aten.bitwise_not.out",
    "aten.bitwise_and.Scalar",
    "aten.bitwise_and_.Scalar",
    "aten.bitwise_and.Scalar_out",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_and_.Tensor",
    "aten.bitwise_and.Tensor_out",
    "aten.bitwise_and.Scalar_Tensor",
    "aten.bitwise_or.Scalar",
    "aten.bitwise_or_.Scalar",
    "aten.bitwise_or.Scalar_out",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_or_.Tensor",
    "aten.bitwise_or.Tensor_out",
    "aten.bitwise_or.Scalar_Tensor",
    "aten.bitwise_xor.Scalar",
    "aten.bitwise_xor_.Scalar",
    "aten.bitwise_xor.Scalar_out",
    "aten.bitwise_xor.Tensor",
    "aten.bitwise_xor_.Tensor",
    "aten.bitwise_xor.Tensor_out",
    "aten.bitwise_xor.Scalar_Tensor",
    "aten.bitwise_left_shift.Tensor",
    "aten.bitwise_left_shift_.Tensor",
    "aten.bitwise_left_shift.Tensor_out",
    "aten.bitwise_left_shift.Tensor_Scalar",
    "aten.bitwise_left_shift_.Tensor_Scalar",
    "aten.bitwise_left_shift.Tensor_Scalar_out",
    "aten.bitwise_left_shift.Scalar_Tensor",
    "aten.bitwise_right_shift.Tensor",
    "aten.bitwise_right_shift_.Tensor",
    "aten.bitwise_right_shift.Tensor_out",
    "aten.bitwise_right_shift.Tensor_Scalar",
    "aten.bitwise_right_shift_.Tensor_Scalar",
    "aten.bitwise_right_shift.Tensor_Scalar_out",
    "aten.bitwise_right_shift.Scalar_Tensor",
    "aten.ceil.default",
    "aten.ceil_.default",
    "aten.ceil.out",
    "aten.clamp.default",
    "aten.clamp_.default",
    "aten.clamp.out",
    "aten.clip.default",
    "aten.clip_.default",
    "aten.clip.out",
    "aten.conj_physical.default",
    "aten.conj_physical_.default",
    "aten.conj_physical.out",
    "aten.copy_sign.Tensor",
    "aten.copy_sign_.Tensor",
    "aten.copy_sign.out",
    "aten.copy_sign.Scalar",
    "aten.copy_sign_.Scalar",
    "aten.copy_sign.Scalar_out",
    "aten.cos.default",
    "aten.cos_.default",
    "aten.cos.out",
    "aten.cosh.default",
    "aten.cosh_.default",
    "aten.cosh.out",
    "aten.deg2rad.default",
    "aten.deg2rad_.default",
    "aten.deg2rad.out",
    "aten.div.Tensor",
    "aten.div_.Tensor",
    "aten.div.out",
    "aten.div.Tensor_mode",
    "aten.div_.Tensor_mode",
    "aten.div.out_mode",
    "aten.digamma.default",
    "aten.digamma_.default",
    "aten.digamma.out",
    "aten.erf.default",
    "aten.erf_.default",
    "aten.erf.out",
    "aten.erfc.default",
    "aten.erfc_.default",
    "aten.erfc.out",
    "aten.erfinv.default",
    "aten.erfinv_.default",
    "aten.erfinv.out",
    "aten.exp.default",
    "aten.exp_.default",
    "aten.exp.out",
    "aten.exp2.default",
    "aten.exp2_.default",
    "aten.exp2.out",
    "aten.expm1.default",
    "aten.expm1_.default",
    "aten.expm1.out",
    "aten.eq.Tensor",
    "aten.float_power.Tensor_Tensor",
    "aten.float_power.Tensor_Tensor_out",
    "aten.float_power.Scalar",
    "aten.float_power.Scalar_out",
    "aten.float_power.Tensor_Scalar",
    "aten.float_power.Tensor_Scalar_out",
    "aten.float_power_.Scalar",
    "aten.float_power_.Tensor",
    "aten.floor.default",
    "aten.floor_.default",
    "aten.floor.out",
    "aten.fmod.Tensor",
    "aten.fmod_.Tensor",
    "aten.fmod.Tensor_out",
    "aten.fmod.Scalar",
    "aten.fmod_.Scalar",
    "aten.fmod.Scalar_out",
    "aten.frac.default",
    "aten.frac_.default",
    "aten.frac.out",
    "aten.gelu.default",
    "aten.hypot.default",
    "aten.hypot_.default",
    "aten.hypot.out",
    "aten.i0.default",
    "aten.i0_.default",
    "aten.i0.out",
    "aten.igamma.default",
    "aten.igamma_.default",
    "aten.igamma.out",
    "aten.igammac.default",
    "aten.igammac_.default",
    "aten.igammac.out",
    "aten.isnan.default",
    "aten.le.Tensor",
    "aten.ldexp.default",
    "aten.ldexp_.default",
    "aten.ldexp.out",
    "aten.lerp.Scalar",
    "aten.lerp.Tensor",
    "aten.lerp_.Scalar",
    "aten.lerp_.Tensor",
    "aten.lerp.Scalar_out",
    "aten.lerp.Tensor_out",
    "aten.lgamma.default",
    "aten.lgamma_.default",
    "aten.lgamma.out",
    "aten.log.default",
    "aten.log_.default",
    "aten.log.out",
    "aten.log10.default",
    "aten.log10_.default",
    "aten.log10.out",
    "aten.log1p.default",
    "aten.log1p_.default",
    "aten.log1p.out",
    "aten.log2.default",
    "aten.log2_.default",
    "aten.log2.out",
    "aten.logaddexp.default",
    "aten.logaddexp.out",
    "aten.logaddexp2.default",
    "aten.logaddexp2.out",
    "aten.logical_and.default",
    "aten.logical_and_.default",
    "aten.logical_and.out",
    "aten.logical_not.default",
    "aten.logical_not_.default",
    "aten.logical_not.out",
    "aten.logical_or.default",
    "aten.logical_or_.default",
    "aten.logical_or.out",
    "aten.logical_xor.default",
    "aten.logical_xor_.default",
    "aten.logical_xor.out",
    "aten.logit.default",
    "aten.logit_.default",
    "aten.logit.out",
    "aten.mul.Scalar",
    "aten.mul_.Scalar",
    "aten.mul.Tensor",
    "aten.mul.out",
    "aten.mul_.Tensor",
    "aten.mvlgamma.default",
    "aten.mvlgamma_.default",
    "aten.mvlgamma.out",
    "aten.nan_to_num.default",
    "aten.nan_to_num_.default",
    "aten.nan_to_num.out",
    "aten.ne.Scalar",
    "aten.neg.default",
    "aten.neg_.default",
    "aten.neg.out",
    "aten.nextafter.default",
    "aten.nextafter_.default",
    "aten.nextafter.out",
    "aten.polygamma.default",
    "aten.polygamma_.default",
    "aten.polygamma.out",
    "aten.positive.default",
    "aten.pow.Tensor_Tensor",
    "aten.pow.Tensor_Tensor_out",
    "aten.pow.Scalar",
    "aten.pow.Scalar_out",
    "aten.pow.Tensor_Scalar",
    "aten.pow.Tensor_Scalar_out",
    "aten.pow_.Tensor",
    "aten.pow_.Scalar",
    "aten.red2deg.default",
    "aten.red2deg_.default",
    "aten.red2deg.out",
    "aten.reciprocal.default",
    "aten.reciprocal_.default",
    "aten.reciprocal.out",
    "aten.remainder.Scalar",
    "aten.remainder_.Scalar",
    "aten.remainder.Scalar_out",
    "aten.remainder.Tensor",
    "aten.remainder_.Tensor",
    "aten.remainder.Tensor_out",
    "aten.remainder.Scalar_Tensor",
    "aten.round.default",
    "aten.round_.default",
    "aten.round.out",
    "aten.round.decimals",
    "aten.round_.decimals",
    "aten.round.decimals_out",
    "aten.rsqrt.default",
    "aten.rsqrt_.default",
    "aten.rsqrt.out",
    "aten.relu.default",
    "aten.relu_.default",
    "aten.sigmoid.default",
    "aten.sigmoid_.default",
    "aten.sigmoid.out",
    "aten.sign.default",
    "aten.sign_.default",
    "aten.sign.out",
    "aten.sgn.default",
    "aten.sgn_.default",
    "aten.sgn.out",
    "aten.signbit.default",
    "aten.signbit.out",
    "aten.sin.default",
    "aten.sin_.default",
    "aten.sin.out",
    "aten.sinc.default",
    "aten.sinc_.default",
    "aten.sinc.out",
    "aten.sinh.default",
    "aten.sinh_.default",
    "aten.sinh.out",
    "aten.sqrt.default",
    "aten.sqrt_.default",
    "aten.sqrt.out",
    "aten.square.default",
    "aten.square_.default",
    "aten.square.out",
    "aten.sub.Tensor",
    "aten.sub_.Tensor",
    "aten.sub.Scalar",
    "aten.sub_.Scalar",
    "aten.sub.out",
    "aten.tan.default",
    "aten.tan_.default",
    "aten.tan.out",
    "aten.tanh.default",
    "aten.tanh_.default",
    "aten.tanh.out",
    "aten.trunc.default",
    "aten.trunc_.default",
    "aten.trunc.out",
    "aten.xlogy.Tensor",
    "aten.xlogy_.Tensor",
    "aten.xlogy.Scalar_self",
    "aten.xlogy.Scalar_other",
    "aten.xlogy_.Scalar_other",
    "aten.xlogy.OutTensor",
    "aten.xlogy.OutScalar_Self",
    "aten.xlogy_.OutScalar_Other",
    # backward point-wise ops
    "aten.threshold_backward.default",
    "aten.gelu_backward.default",
    "aten.tanh_backward.default",
    "aten.sgn.default",
    "aten.neg.default",
    "aten._softmax_backward_data.default",
]


def _replace_char_in_str(string: str, new_char: str, idx: int) -> str:
    return string[:idx] + new_char + string[idx + 1 :]


def _inplace_rewrap_schema_suggestion(
    suggestion: OpSchema, input_schema: OpSchema
) -> None:
    suggestion_args_spec = suggestion.args_spec
    new_arg_schema: List[object] = []
    idx_of_args_spec = 0
    for arg in input_schema.args_schema:
        if isinstance(arg, DTensorSpec):
            new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
            idx_of_args_spec += 1
        else:
            new_arg_schema.append(arg)
    suggestion.args_schema = tuple(new_arg_schema)
    suggestion.kwargs_schema = input_schema.kwargs_schema


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
    einop_schema = OpSchema(input_specs, {})
    output_sharding = einop_rule(fmt, einop_schema, linearity=linearity)
    if (
        output_sharding.output_spec is None
        and output_sharding.schema_suggestions is not None
    ):
        # sharding propagation failed, with reshard suggetion only have tensor specs,
        # we will need to include back all non-tensor args/kwargs
        suggested_schema = output_sharding.schema_suggestions[0]
        _inplace_rewrap_schema_suggestion(suggested_schema, op_schema)

    return output_sharding


for op in pointwise_ops:
    DTensor._op_to_rules[op] = pointwise_rule


@register_prop_rule("aten._softmax_backward_data.default")
def softmax_bwd_rule(op_schema: OpSchema) -> OutputSharding:
    input_specs = cast(List[DTensorSpec], op_schema.args_spec)
    ops_dim_map = pytree.tree_map(lambda spec: spec.dim_map, input_specs)
    softmax_dim = cast(int, op_schema.args_schema[len(op_schema.args_spec)])
    ops_dim_map = list(zip(*ops_dim_map))
    if softmax_dim < len(ops_dim_map) and 0 in ops_dim_map[softmax_dim]:
        schema_suggestion = OpSchema(
            tuple(
                pytree.tree_map(
                    lambda spec: DTensorSpec(
                        spec.mesh, [Replicate()], spec.shape, ndim=spec.ndim
                    ),
                    input_specs,
                )
            ),
            {},
        )
        failed_reason = "Cannot run _softmax_backward_data on batch dim, need to replicate the tensor."
        _inplace_rewrap_schema_suggestion(schema_suggestion, op_schema)
        return OutputSharding(None, [schema_suggestion], failed_reason)
    else:
        return pointwise_rule(op_schema)
