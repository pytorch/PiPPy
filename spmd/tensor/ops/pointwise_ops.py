# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.math_ops import einop_rule

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
#     "mul",
#     "multiply",
#     "quantized_batch_norm",
#     "quantized_max_pool1d",
#     "quantized_max_pool2d",
#     "real",  # complex data type only
#     "tanh",
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
    # TODO: handle pointwise ops that have multiple outputs
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
