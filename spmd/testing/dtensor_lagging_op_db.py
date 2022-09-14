# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
from torch.testing._internal.common_methods_invocations import op_db, OpInfo

# Generated from test/gen_dtensor_op_db.py via
# python spmd/test/gen_dtensor_lagging_op_db.py > spmd/test/dtensor_lagging_op_db.py
#
# This approach is copied from functorch:
# People add new OpInfos to PyTorch all the time.
# We want them to be able to add OpInfos without breaking our CI.
# To achieve this, we keep our OpInfo library behind that of Pytorch's and
# we periodically update our OpInfo library by regenerating this file
_dtensor_lagging_meta = {
    ("H", ""),
    ("T", ""),
    ("__getitem__", ""),
    ("__radd__", ""),
    ("__rand__", ""),
    ("__rdiv__", ""),
    ("__rmatmul__", ""),
    ("__rmod__", ""),
    ("__rmul__", ""),
    ("__ror__", ""),
    ("__rpow__", ""),
    ("__rsub__", ""),
    ("__rxor__", ""),
    ("_masked.amax", ""),
    ("_masked.amin", ""),
    ("_masked.argmax", ""),
    ("_masked.argmin", ""),
    ("_masked.cumprod", ""),
    ("_masked.cumsum", ""),
    ("_masked.log_softmax", ""),
    ("_masked.logaddexp", ""),
    ("_masked.logsumexp", ""),
    ("_masked.mean", ""),
    ("_masked.median", ""),
    ("_masked.norm", ""),
    ("_masked.normalize", ""),
    ("_masked.prod", ""),
    ("_masked.softmax", ""),
    ("_masked.softmin", ""),
    ("_masked.std", ""),
    ("_masked.sum", ""),
    ("_masked.var", ""),
    ("abs", ""),
    ("acos", ""),
    ("acosh", ""),
    ("add", ""),
    ("addbmm", ""),
    ("addcdiv", ""),
    ("addcmul", ""),
    ("addmm", ""),
    ("addmm", "decomposed"),
    ("addmv", ""),
    ("addr", ""),
    ("all", ""),
    ("allclose", ""),
    ("amax", ""),
    ("amin", ""),
    ("aminmax", ""),
    ("angle", ""),
    ("any", ""),
    ("arange", ""),
    ("argmax", ""),
    ("argmin", ""),
    ("argsort", ""),
    ("argwhere", ""),
    ("as_strided", ""),
    ("as_strided_scatter", ""),
    ("asin", ""),
    ("asinh", ""),
    ("atan", ""),
    ("atan2", ""),
    ("atanh", ""),
    ("atleast_1d", ""),
    ("atleast_2d", ""),
    ("atleast_3d", ""),
    ("baddbmm", ""),
    ("bernoulli", ""),
    ("bfloat16", ""),
    ("bincount", ""),
    ("bitwise_and", ""),
    ("bitwise_left_shift", ""),
    ("bitwise_not", ""),
    ("bitwise_or", ""),
    ("bitwise_right_shift", ""),
    ("bitwise_xor", ""),
    ("block_diag", ""),
    ("bmm", ""),
    ("bool", ""),
    ("broadcast_shapes", ""),
    ("broadcast_tensors", ""),
    ("broadcast_to", ""),
    ("bucketize", ""),
    ("byte", ""),
    ("cartesian_prod", ""),
    ("cat", ""),
    ("cdist", ""),
    ("ceil", ""),
    ("chalf", ""),
    ("char", ""),
    ("cholesky", ""),
    ("cholesky_inverse", ""),
    ("cholesky_solve", ""),
    ("chunk", ""),
    ("clamp", ""),
    ("clamp_max", ""),
    ("clamp_min", ""),
    ("clone", ""),
    ("column_stack", ""),
    ("combinations", ""),
    ("complex", ""),
    ("conj", ""),
    ("conj_physical", ""),
    ("constant_pad_nd", ""),
    ("contiguous", ""),
    ("copysign", ""),
    ("corrcoef", ""),
    ("cos", ""),
    ("cosh", ""),
    ("count_nonzero", ""),
    ("cov", ""),
    ("cross", ""),
    ("cummax", ""),
    ("cummin", ""),
    ("cumprod", ""),
    ("cumsum", ""),
    ("cumulative_trapezoid", ""),
    ("deg2rad", ""),
    ("diag", ""),
    ("diag_embed", ""),
    ("diagflat", ""),
    ("diagonal", ""),
    ("diagonal_scatter", ""),
    ("diff", ""),
    ("digamma", ""),
    ("dist", ""),
    ("div", "floor_rounding"),
    ("div", "no_rounding_mode"),
    ("div", "trunc_rounding"),
    ("dot", ""),
    ("double", ""),
    ("dsplit", ""),
    ("dstack", ""),
    ("einsum", ""),
    ("empty", ""),
    ("empty_like", ""),
    ("eq", ""),
    ("equal", ""),
    ("erf", ""),
    ("erfc", ""),
    ("erfinv", ""),
    ("exp", ""),
    ("exp2", ""),
    ("expand", ""),
    ("expand_as", ""),
    ("expm1", ""),
    ("eye", ""),
    ("fft.fft", ""),
    ("fft.fft2", ""),
    ("fft.fftn", ""),
    ("fft.fftshift", ""),
    ("fft.hfft", ""),
    ("fft.hfft2", ""),
    ("fft.hfftn", ""),
    ("fft.ifft", ""),
    ("fft.ifft2", ""),
    ("fft.ifftn", ""),
    ("fft.ifftshift", ""),
    ("fft.ihfft", ""),
    ("fft.ihfft2", ""),
    ("fft.ihfftn", ""),
    ("fft.irfft", ""),
    ("fft.irfft2", ""),
    ("fft.irfftn", ""),
    ("fft.rfft", ""),
    ("fft.rfft2", ""),
    ("fft.rfftn", ""),
    ("fill", ""),
    ("flatten", ""),
    ("flip", ""),
    ("fliplr", ""),
    ("flipud", ""),
    ("float", ""),
    ("float_power", ""),
    ("floor", ""),
    ("floor_divide", ""),
    ("fmax", ""),
    ("fmin", ""),
    ("fmod", ""),
    ("frac", ""),
    ("frexp", ""),
    ("full_like", ""),
    ("gather", ""),
    ("gcd", ""),
    ("ge", ""),
    ("geqrf", ""),
    ("gradient", ""),
    ("gt", ""),
    ("half", ""),
    ("heaviside", ""),
    ("histc", ""),
    ("histogram", ""),
    ("histogramdd", ""),
    ("hsplit", ""),
    ("hstack", ""),
    ("hypot", ""),
    ("i0", ""),
    ("igamma", ""),
    ("igammac", ""),
    ("imag", ""),
    ("index_add", ""),
    ("index_copy", ""),
    ("index_fill", ""),
    ("index_put", ""),
    ("index_reduce", ""),
    ("index_select", ""),
    ("inner", ""),
    ("int", ""),
    ("isclose", ""),
    ("isfinite", ""),
    ("isin", ""),
    ("isinf", ""),
    ("isnan", ""),
    ("isneginf", ""),
    ("isposinf", ""),
    ("isreal", ""),
    ("istft", ""),
    ("jiterator_2inputs_2outputs", ""),
    ("jiterator_4inputs_with_extra_args", ""),
    ("jiterator_binary", ""),
    ("jiterator_binary_return_by_ref", ""),
    ("jiterator_unary", ""),
    ("kron", ""),
    ("kthvalue", ""),
    ("lcm", ""),
    ("ldexp", ""),
    ("le", ""),
    ("lerp", ""),
    ("lgamma", ""),
    ("linalg.cholesky", ""),
    ("linalg.cholesky_ex", ""),
    ("linalg.cond", ""),
    ("linalg.cross", ""),
    ("linalg.det", ""),
    ("linalg.det", "singular"),
    ("linalg.eig", ""),
    ("linalg.eigh", ""),
    ("linalg.eigvals", ""),
    ("linalg.eigvalsh", ""),
    ("linalg.householder_product", ""),
    ("linalg.inv", ""),
    ("linalg.inv_ex", ""),
    ("linalg.ldl_factor", ""),
    ("linalg.ldl_factor_ex", ""),
    ("linalg.ldl_solve", ""),
    ("linalg.lstsq", ""),
    ("linalg.lstsq", "grad_oriented"),
    ("linalg.lu", ""),
    ("linalg.lu_factor", ""),
    ("linalg.lu_factor_ex", ""),
    ("linalg.lu_solve", ""),
    ("linalg.matrix_norm", ""),
    ("linalg.matrix_power", ""),
    ("linalg.matrix_rank", ""),
    ("linalg.matrix_rank", "hermitian"),
    ("linalg.multi_dot", ""),
    ("linalg.norm", ""),
    ("linalg.norm", "subgradients_at_zero"),
    ("linalg.pinv", ""),
    ("linalg.pinv", "hermitian"),
    ("linalg.pinv", "singular"),
    ("linalg.qr", ""),
    ("linalg.slogdet", ""),
    ("linalg.solve", ""),
    ("linalg.solve_ex", ""),
    ("linalg.solve_triangular", ""),
    ("linalg.svd", ""),
    ("linalg.svdvals", ""),
    ("linalg.tensorinv", ""),
    ("linalg.tensorsolve", ""),
    ("linalg.vander", ""),
    ("linalg.vecdot", ""),
    ("linalg.vector_norm", ""),
    ("linspace", ""),
    ("log", ""),
    ("log10", ""),
    ("log1p", ""),
    ("log2", ""),
    ("log_softmax", ""),
    ("log_softmax", "dtype"),
    ("logaddexp", ""),
    ("logaddexp2", ""),
    ("logcumsumexp", ""),
    ("logdet", ""),
    ("logical_and", ""),
    ("logical_not", ""),
    ("logical_or", ""),
    ("logical_xor", ""),
    ("logit", ""),
    ("logspace", ""),
    ("logsumexp", ""),
    ("long", ""),
    ("lt", ""),
    ("lu", ""),
    ("lu_solve", ""),
    ("lu_unpack", ""),
    ("mH", ""),
    ("mT", ""),
    ("masked_fill", ""),
    ("masked_scatter", ""),
    ("masked_select", ""),
    ("matmul", ""),
    ("matrix_exp", ""),
    ("max", "binary"),
    ("max", "reduction_no_dim"),
    ("max", "reduction_with_dim"),
    ("maximum", ""),
    ("mean", ""),
    ("median", ""),
    ("meshgrid", "list_of_tensors"),
    ("meshgrid", "variadic_tensors"),
    ("min", "binary"),
    ("min", "reduction_no_dim"),
    ("min", "reduction_with_dim"),
    ("minimum", ""),
    ("mm", ""),
    ("mode", ""),
    ("movedim", ""),
    ("msort", ""),
    ("mul", ""),
    ("multinomial", ""),
    ("mv", ""),
    ("mvlgamma", "mvlgamma_p_1"),
    ("mvlgamma", "mvlgamma_p_3"),
    ("mvlgamma", "mvlgamma_p_5"),
    ("nan_to_num", ""),
    ("nanmean", ""),
    ("nanmedian", ""),
    ("nanquantile", ""),
    ("nansum", ""),
    ("narrow", ""),
    ("native_layer_norm", ""),
    ("ne", ""),
    ("neg", ""),
    ("new_empty", ""),
    ("new_empty_strided", ""),
    ("new_full", ""),
    ("new_ones", ""),
    ("new_zeros", ""),
    ("nextafter", ""),
    ("nn.functional.adaptive_avg_pool1d", ""),
    ("nn.functional.adaptive_avg_pool2d", ""),
    ("nn.functional.adaptive_avg_pool3d", ""),
    ("nn.functional.adaptive_max_pool1d", ""),
    ("nn.functional.adaptive_max_pool2d", ""),
    ("nn.functional.adaptive_max_pool3d", ""),
    ("nn.functional.avg_pool1d", ""),
    ("nn.functional.avg_pool2d", ""),
    ("nn.functional.avg_pool3d", ""),
    ("nn.functional.batch_norm", ""),
    ("nn.functional.batch_norm", "without_cudnn"),
    ("nn.functional.bilinear", ""),
    ("nn.functional.binary_cross_entropy", ""),
    ("nn.functional.binary_cross_entropy_with_logits", ""),
    ("nn.functional.celu", ""),
    ("nn.functional.conv1d", ""),
    ("nn.functional.conv2d", ""),
    ("nn.functional.conv_transpose1d", ""),
    ("nn.functional.conv_transpose2d", ""),
    ("nn.functional.conv_transpose3d", ""),
    ("nn.functional.cosine_embedding_loss", ""),
    ("nn.functional.cosine_similarity", ""),
    ("nn.functional.cross_entropy", ""),
    ("nn.functional.ctc_loss", ""),
    ("nn.functional.dropout", ""),
    ("nn.functional.dropout2d", ""),
    ("nn.functional.dropout3d", ""),
    ("nn.functional.elu", ""),
    ("nn.functional.embedding", ""),
    ("nn.functional.embedding_bag", ""),
    ("nn.functional.feature_alpha_dropout", "with_train"),
    ("nn.functional.feature_alpha_dropout", "without_train"),
    ("nn.functional.fractional_max_pool2d", ""),
    ("nn.functional.fractional_max_pool3d", ""),
    ("nn.functional.gaussian_nll_loss", ""),
    ("nn.functional.gelu", ""),
    ("nn.functional.glu", ""),
    ("nn.functional.grid_sample", ""),
    ("nn.functional.group_norm", ""),
    ("nn.functional.hardshrink", ""),
    ("nn.functional.hardsigmoid", ""),
    ("nn.functional.hardswish", ""),
    ("nn.functional.hardtanh", ""),
    ("nn.functional.hinge_embedding_loss", ""),
    ("nn.functional.huber_loss", ""),
    ("nn.functional.instance_norm", ""),
    ("nn.functional.interpolate", "area"),
    ("nn.functional.interpolate", "bicubic"),
    ("nn.functional.interpolate", "bilinear"),
    ("nn.functional.interpolate", "linear"),
    ("nn.functional.interpolate", "nearest"),
    ("nn.functional.interpolate", "trilinear"),
    ("nn.functional.kl_div", ""),
    ("nn.functional.l1_loss", ""),
    ("nn.functional.layer_norm", ""),
    ("nn.functional.leaky_relu", ""),
    ("nn.functional.linear", ""),
    ("nn.functional.local_response_norm", ""),
    ("nn.functional.logsigmoid", ""),
    ("nn.functional.margin_ranking_loss", ""),
    ("nn.functional.max_pool1d", ""),
    ("nn.functional.max_pool2d", ""),
    ("nn.functional.max_pool3d", ""),
    ("nn.functional.max_unpool1d", ""),
    ("nn.functional.max_unpool1d", "grad"),
    ("nn.functional.max_unpool2d", ""),
    ("nn.functional.max_unpool2d", "grad"),
    ("nn.functional.max_unpool3d", ""),
    ("nn.functional.max_unpool3d", "grad"),
    ("nn.functional.mish", ""),
    ("nn.functional.mse_loss", ""),
    ("nn.functional.multi_margin_loss", ""),
    ("nn.functional.multilabel_margin_loss", ""),
    ("nn.functional.multilabel_soft_margin_loss", ""),
    ("nn.functional.nll_loss", ""),
    ("nn.functional.normalize", ""),
    ("nn.functional.one_hot", ""),
    ("nn.functional.pad", "circular"),
    ("nn.functional.pad", "constant"),
    ("nn.functional.pad", "reflect"),
    ("nn.functional.pad", "replicate"),
    ("nn.functional.pairwise_distance", ""),
    ("nn.functional.pdist", ""),
    ("nn.functional.pixel_shuffle", ""),
    ("nn.functional.pixel_unshuffle", ""),
    ("nn.functional.poisson_nll_loss", ""),
    ("nn.functional.prelu", ""),
    ("nn.functional.relu", ""),
    ("nn.functional.relu6", ""),
    ("nn.functional.rrelu", ""),
    ("nn.functional.selu", ""),
    ("nn.functional.silu", ""),
    ("nn.functional.silu", "complex"),
    ("nn.functional.smooth_l1_loss", ""),
    ("nn.functional.soft_margin_loss", ""),
    ("nn.functional.softmin", ""),
    ("nn.functional.softmin", "with_dtype"),
    ("nn.functional.softplus", ""),
    ("nn.functional.softshrink", ""),
    ("nn.functional.softsign", ""),
    ("nn.functional.tanhshrink", ""),
    ("nn.functional.threshold", ""),
    ("nn.functional.triplet_margin_loss", ""),
    ("nn.functional.triplet_margin_with_distance_loss", ""),
    ("nn.functional.unfold", ""),
    ("nn.functional.upsample_bilinear", ""),
    ("nn.functional.upsample_nearest", ""),
    ("nonzero", ""),
    ("norm", ""),
    ("norm", "fro"),
    ("norm", "inf"),
    ("norm", "nuc"),
    ("normal", ""),
    ("normal", "number_mean"),
    ("ones_like", ""),
    ("ormqr", ""),
    ("outer", ""),
    ("pca_lowrank", ""),
    ("permute", ""),
    ("pinverse", ""),
    ("polar", ""),
    ("polygamma", "polygamma_n_0"),
    ("polygamma", "polygamma_n_1"),
    ("polygamma", "polygamma_n_2"),
    ("polygamma", "polygamma_n_3"),
    ("polygamma", "polygamma_n_4"),
    ("positive", ""),
    ("pow", ""),
    ("prod", ""),
    ("put", ""),
    ("qr", ""),
    ("quantile", ""),
    ("rad2deg", ""),
    ("rand_like", ""),
    ("randint_like", ""),
    ("randn_like", ""),
    ("ravel", ""),
    ("real", ""),
    ("reciprocal", ""),
    ("remainder", ""),
    ("renorm", ""),
    ("repeat", ""),
    ("repeat_interleave", ""),
    ("reshape", ""),
    ("reshape_as", ""),
    ("resize_", ""),
    ("resize_as_", ""),
    ("resolve_conj", ""),
    ("resolve_neg", ""),
    ("roll", ""),
    ("rot90", ""),
    ("round", ""),
    ("round", "decimals_0"),
    ("round", "decimals_3"),
    ("round", "decimals_neg_3"),
    ("rsqrt", ""),
    ("rsub", ""),
    ("scatter", ""),
    ("scatter_add", ""),
    ("scatter_reduce", "amax"),
    ("scatter_reduce", "amin"),
    ("scatter_reduce", "mean"),
    ("scatter_reduce", "prod"),
    ("scatter_reduce", "sum"),
    ("searchsorted", ""),
    ("segment_reduce", "lengths"),
    ("segment_reduce", "offsets"),
    ("select", ""),
    ("select_scatter", ""),
    ("sgn", ""),
    ("short", ""),
    ("sigmoid", ""),
    ("sign", ""),
    ("signbit", ""),
    ("sin", ""),
    ("sinc", ""),
    ("sinh", ""),
    ("slice_scatter", ""),
    ("softmax", ""),
    ("softmax", "with_dtype"),
    ("sort", ""),
    ("sparse.sampled_addmm", ""),
    ("special.airy_ai", ""),
    ("special.bessel_j0", ""),
    ("special.bessel_j1", ""),
    ("special.bessel_y0", ""),
    ("special.bessel_y1", ""),
    ("special.chebyshev_polynomial_t", ""),
    ("special.chebyshev_polynomial_u", ""),
    ("special.chebyshev_polynomial_v", ""),
    ("special.chebyshev_polynomial_w", ""),
    ("special.entr", ""),
    ("special.erfcx", ""),
    ("special.hermite_polynomial_h", ""),
    ("special.hermite_polynomial_he", ""),
    ("special.i0e", ""),
    ("special.i1", ""),
    ("special.i1e", ""),
    ("special.laguerre_polynomial_l", ""),
    ("special.legendre_polynomial_p", ""),
    ("special.log_ndtr", ""),
    ("special.modified_bessel_i0", ""),
    ("special.modified_bessel_i1", ""),
    ("special.modified_bessel_k0", ""),
    ("special.modified_bessel_k1", ""),
    ("special.ndtr", ""),
    ("special.ndtri", ""),
    ("special.polygamma", "special_polygamma_n_0"),
    ("special.scaled_modified_bessel_k0", ""),
    ("special.scaled_modified_bessel_k1", ""),
    ("special.shifted_chebyshev_polynomial_t", ""),
    ("special.shifted_chebyshev_polynomial_u", ""),
    ("special.shifted_chebyshev_polynomial_v", ""),
    ("special.shifted_chebyshev_polynomial_w", ""),
    ("special.spherical_bessel_j0", ""),
    ("special.xlog1py", ""),
    ("special.zeta", ""),
    ("split", ""),
    ("split", "list_args"),
    ("split_with_sizes", ""),
    ("sqrt", ""),
    ("square", ""),
    ("squeeze", ""),
    ("stack", ""),
    ("std", ""),
    ("std_mean", ""),
    ("stft", ""),
    ("sub", ""),
    ("sum", ""),
    ("sum_to_size", ""),
    ("svd", ""),
    ("svd_lowrank", ""),
    ("symeig", ""),
    ("t", ""),
    ("take", ""),
    ("take_along_dim", ""),
    ("tan", ""),
    ("tanh", ""),
    ("tensor_split", ""),
    ("tensordot", ""),
    ("tile", ""),
    ("to_sparse", ""),
    ("topk", ""),
    ("trace", ""),
    ("transpose", ""),
    ("trapezoid", ""),
    ("trapz", ""),
    ("triangular_solve", ""),
    ("tril", ""),
    ("tril_indices", ""),
    ("triu", ""),
    ("triu_indices", ""),
    ("true_divide", ""),
    ("trunc", ""),
    ("unbind", ""),
    ("unflatten", ""),
    ("unfold", ""),
    ("unique", ""),
    ("unique_consecutive", ""),
    ("unsqueeze", ""),
    ("var", ""),
    ("var_mean", ""),
    ("vdot", ""),
    ("view", ""),
    ("view_as", ""),
    ("view_as_complex", ""),
    ("view_as_real", ""),
    ("vsplit", ""),
    ("vstack", ""),
    ("where", ""),
    ("xlogy", ""),
    ("zero_", ""),
    ("zeros_like", ""),
}


def in_dtensor_lagging_op_db(opinfo: OpInfo) -> bool:
    return (opinfo.name, opinfo.variant_test_name) in _dtensor_lagging_meta


dtensor_lagging_op_db: List[OpInfo] = [
    opinfo for opinfo in op_db if in_dtensor_lagging_op_db(opinfo)
]
