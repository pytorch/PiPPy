# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["module: distributed"]

import torch
import sys
import unittest
import warnings

from torch.overrides import resolve_name
from torch.utils._pytree import tree_flatten, tree_map
from torch.testing._internal.common_utils import (
    suppress_warnings,
    TEST_WITH_ASAN,
    run_tests,
)
import torch.distributed as dist
from torch.testing._internal.common_device_type import (
    ops,
    instantiate_device_type_tests,
)
import torch.testing._internal.common_methods_invocations as common_ops
from torch.testing._internal.common_methods_invocations import DecorateInfo

from spmd import DTensor, DeviceMesh, Replicate
from spmd.testing.dtensor_lagging_op_db import dtensor_lagging_op_db
from spmd.testing.common_utils import (
    DistTensorTestBase,
    TEST_SKIPS,
    DTensorConverter,
)


DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DEVICES = 4

# rewrite common size variables to sth can be sharded evenly
# we can enable uneven shards later, but need to adjust more on
# sample inputs (i.e. view/reshape need to adjust shape size as well)
common_ops.L = 24
common_ops.M = 12
common_ops.S = 4
common_ops.XS = 2


def assert_ref_dtensor_equal(test_case, dtensor_rs, rs):
    flat_dtensor_rs, _ = tree_flatten(dtensor_rs)
    flat_rs, _ = tree_flatten(rs)
    test_case.assertEqual(len(flat_dtensor_rs), len(flat_rs))
    for dtensor_r, r in zip(flat_dtensor_rs, flat_rs):

        if not isinstance(r, torch.Tensor):
            continue

        test_case.assertIsInstance(dtensor_r, torch.Tensor)
        test_case.assertEqual(
            dtensor_r.shape,
            r.shape,
            f"Shape mismatch! original shape:{r.shape}, dtensor shape: {dtensor_r.shape}",
        )
        test_case.assertEqual(
            dtensor_r.requires_grad,
            r.requires_grad,
            "op result requires_grad mismatch!"
            f"original requires_grad: {r.requires_grad}, "
            f"dtensor requires_grad: {dtensor_r.requires_grad}",
        )

        test_case.assertEqual(dtensor_r.to_local(), r)


# Copied from functorch
def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)


def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, False)


def skipOps(test_case_name, base_test_name, to_skip):
    all_opinfos = dtensor_lagging_op_db
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == op_name and o.variant_test_name == variant_name
        ]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(
                    unittest.expectedFailure,
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(
                    unittest.skip("Skipped!"),
                    test_case_name,
                    base_test_name,
                    device_type=device_type,
                    dtypes=dtypes,
                )
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn

    return wrapped


# Re-generate this failed list, turn on dry_run of the below func
# check_dtensor_func(self, test, op, dry_run=True), then run sth
# like python test/spmd/tensor/test_dtensor_ops.py > failed.expect
dtensor_fails = {
    # these sometimes pass and sometimes fail
    # we need to remove many of them from list once op
    # get full support with varying sharding specs
    xfail("__getitem__"),
    xfail("__rdiv__"),
    xfail("__rmod__"),
    xfail("__rpow__"),
    xfail("__rsub__"),
    xfail("_masked.amax"),
    xfail("_masked.amin"),
    xfail("_masked.argmax"),
    xfail("_masked.argmin"),
    xfail("_masked.cumprod"),
    xfail("_masked.cumsum"),
    xfail("_masked.log_softmax"),
    xfail("_masked.logaddexp"),
    xfail("_masked.logsumexp"),
    xfail("_masked.median"),
    xfail("_masked.norm"),
    xfail("_masked.prod"),
    xfail("_masked.softmin"),
    xfail("_masked.softmax"),
    xfail("_masked.sum"),
    xfail("acos"),
    xfail("acosh"),
    xfail("add"),
    xfail("addbmm"),
    xfail("addcdiv"),
    xfail("addcmul"),
    xfail("addmv"),
    xfail("addr"),
    xfail("all"),
    xfail("allclose"),
    xfail("amax"),
    xfail("amin"),
    xfail("aminmax"),
    xfail("angle"),
    xfail("any"),
    xfail("arange"),
    xfail("argmax"),
    xfail("argmin"),
    xfail("argsort"),
    xfail("as_strided"),
    xfail("as_strided_scatter"),
    xfail("asin"),
    xfail("asinh"),
    xfail("atan2"),
    xfail("atan"),
    xfail("atanh"),
    xfail("baddbmm"),
    xfail("bernoulli"),
    xfail("block_diag"),
    xfail("bmm"),
    xfail("broadcast_shapes"),
    xfail("bucketize"),
    xfail("cat"),
    xfail("cartesian_prod"),
    xfail("cdist"),
    xfail("ceil"),
    xfail("cholesky"),
    xfail("cholesky_inverse"),
    xfail("cholesky_solve"),
    xfail("chunk"),
    xfail("clamp"),
    xfail("clamp_max"),
    xfail("clamp_min"),
    xfail("column_stack"),
    xfail("combinations"),
    xfail("complex"),
    xfail("constant_pad_nd"),
    xfail("copysign"),
    xfail("corrcoef"),
    xfail("cos"),
    xfail("cosh"),
    xfail("count_nonzero"),
    xfail("cov"),
    xfail("cross"),
    xfail("cummax"),
    xfail("cummin"),
    xfail("cumsum"),
    xfail("cumulative_trapezoid"),
    xfail("deg2rad"),
    xfail("diag"),
    xfail("diag_embed"),
    xfail("diagflat"),
    xfail("diagonal"),
    xfail("diagonal_scatter"),
    xfail("diff"),
    xfail("digamma"),
    xfail("dist"),
    xfail("div", "floor_rounding"),
    xfail("div", "no_rounding_mode"),
    xfail("div", "trunc_rounding"),
    xfail("dot"),
    xfail("dsplit"),
    xfail("dstack"),
    xfail("eig"),
    xfail("einsum"),
    xfail("empty"),
    xfail("empty_like"),
    xfail("eq"),
    xfail("equal"),
    xfail("erf"),
    xfail("erfc"),
    xfail("erfinv"),
    xfail("exp2"),
    xfail("exp"),
    xfail("expm1"),
    xfail("eye"),
    xfail("fft.fft2"),
    xfail("fft.fft"),
    xfail("fft.fftn"),
    xfail("fft.fftshift"),
    xfail("fft.ifft2"),
    xfail("fft.ifft"),
    xfail("fft.ifftshift"),
    xfail("fft.ihfft2"),
    xfail("fft.ihfft"),
    xfail("fft.ihfftn"),
    xfail("fft.irfft2"),
    xfail("fft.irfftn"),
    xfail("fft.rfft2"),
    xfail("fft.rfft"),
    xfail("fft.rfftn"),
    xfail("fill"),
    xfail("flatten"),
    xfail("flip"),
    xfail("fliplr"),
    xfail("flipud"),
    xfail("float_power"),
    xfail("floor"),
    xfail("floor_divide"),
    xfail("fmax"),
    xfail("fmin"),
    xfail("fmod"),
    xfail("frac"),
    xfail("frexp"),
    xfail("full_like"),
    xfail("gather"),
    xfail("ge"),
    xfail("geqrf"),
    xfail("gradient"),
    xfail("gt"),
    xfail("heaviside"),
    xfail("histc"),
    xfail("histogram"),
    xfail("histogramdd"),
    xfail("hsplit"),
    xfail("hstack"),
    xfail("hypot"),
    xfail("i0"),
    xfail("igamma"),
    xfail("igammac"),
    xfail("index_add"),
    xfail("index_copy"),
    xfail("index_fill"),
    xfail("index_put"),
    xfail("index_reduce"),
    xfail("index_select"),
    xfail("isfinite"),
    xfail("isin"),
    xfail("isinf"),
    xfail("isnan"),
    xfail("isneginf"),
    xfail("isposinf"),
    xfail("kron"),
    xfail("kthvalue"),
    xfail("ldexp"),
    xfail("lerp"),
    xfail("lgamma"),
    xfail("linalg.cholesky"),
    xfail("linalg.cholesky_ex"),
    xfail("linalg.cond"),
    xfail("linalg.cross"),
    xfail("linalg.det"),
    xfail("linalg.det", "singular"),
    xfail("linalg.eig"),
    xfail("linalg.eigh"),
    xfail("linalg.eigvals"),
    xfail("linalg.eigvalsh"),
    xfail("linalg.householder_product"),
    xfail("linalg.inv"),
    xfail("linalg.inv_ex"),
    xfail("linalg.ldl_factor"),
    xfail("linalg.ldl_factor_ex"),
    xfail("linalg.ldl_solve"),
    xfail("linalg.lstsq"),
    xfail("linalg.lstsq", "grad_oriented"),
    xfail("linalg.lu"),
    xfail("linalg.lu_factor"),
    xfail("linalg.lu_factor_ex"),
    xfail("linalg.lu_solve"),
    xfail("linalg.matrix_norm"),
    xfail("linalg.matrix_power"),
    xfail("linalg.matrix_rank"),
    xfail("linalg.matrix_rank", "hermitian"),
    xfail("linalg.multi_dot"),
    xfail("linalg.norm"),
    xfail("linalg.norm", "subgradients_at_zero"),
    xfail("linalg.pinv"),
    xfail("linalg.pinv", "hermitian"),
    xfail("linalg.qr"),
    xfail("linalg.slogdet"),
    xfail("linalg.solve"),
    xfail("linalg.solve_ex"),
    xfail("linalg.solve_triangular"),
    xfail("linalg.svd"),
    xfail("linalg.svdvals"),
    xfail("linalg.tensorinv"),
    xfail("linalg.tensorsolve"),
    xfail("linalg.vander"),
    xfail("linalg.vecdot"),
    xfail("linalg.vector_norm"),
    xfail("linspace"),
    xfail("log10"),
    xfail("log1p"),
    xfail("log2"),
    xfail("log"),
    xfail("log_softmax"),
    xfail("log_softmax", "dtype"),
    xfail("logaddexp2"),
    xfail("logaddexp"),
    xfail("logcumsumexp"),
    xfail("logdet"),
    xfail("logical_and"),
    xfail("logical_not"),
    xfail("logical_or"),
    xfail("logical_xor"),
    xfail("logit"),
    xfail("logspace"),
    xfail("logsumexp"),
    xfail("lt"),
    xfail("lu"),
    xfail("lu_solve"),
    xfail("lu_unpack"),
    xfail("masked_fill"),
    xfail("masked_scatter"),
    xfail("masked_select"),
    xfail("matrix_exp"),
    xfail("max", "binary"),
    xfail("max", "reduction_no_dim"),
    xfail("max", "reduction_with_dim"),
    xfail("maximum"),
    xfail("mean"),
    xfail("median"),
    xfail("min", "binary"),
    xfail("min", "reduction_no_dim"),
    xfail("min", "reduction_with_dim"),
    xfail("minimum"),
    xfail("mm"),
    xfail("mode"),
    xfail("msort"),
    xfail("multinomial"),
    xfail("mv"),
    xfail("mvlgamma", "mvlgamma_p_1"),
    xfail("mvlgamma", "mvlgamma_p_3"),
    xfail("mvlgamma", "mvlgamma_p_5"),
    xfail("nan_to_num"),
    xfail("nanmean"),
    xfail("nanmedian"),
    xfail("nanquantile"),
    xfail("nansum"),
    xfail("narrow"),
    xfail("native_layer_norm"),
    xfail("ne"),
    xfail("new_empty"),
    xfail("new_empty_strided"),
    xfail("new_full"),
    xfail("new_ones"),
    xfail("new_zeros"),
    xfail("nextafter"),
    xfail("transpose"),
    xfail("nn.functional.adaptive_avg_pool1d"),
    xfail("nn.functional.adaptive_avg_pool2d"),
    xfail("nn.functional.adaptive_avg_pool3d"),
    xfail("nn.functional.adaptive_max_pool1d"),
    xfail("nn.functional.adaptive_max_pool2d"),
    xfail("nn.functional.adaptive_max_pool3d"),
    xfail("nn.functional.avg_pool1d"),
    xfail("nn.functional.avg_pool2d"),
    xfail("nn.functional.avg_pool3d"),
    xfail("nn.functional.batch_norm"),
    xfail("nn.functional.bilinear"),
    xfail("nn.functional.binary_cross_entropy"),
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    xfail("nn.functional.celu"),
    xfail("nn.functional.conv1d"),
    xfail("nn.functional.conv2d"),
    xfail("nn.functional.conv_transpose1d"),
    xfail("nn.functional.conv_transpose2d"),
    xfail("nn.functional.conv_transpose3d"),
    xfail("nn.functional.cosine_similarity"),
    xfail("nn.functional.cross_entropy"),
    xfail("nn.functional.ctc_loss"),
    xfail("nn.functional.dropout"),
    xfail("nn.functional.dropout2d"),
    xfail("nn.functional.dropout3d"),
    xfail("nn.functional.elu"),
    xfail("nn.functional.fractional_max_pool2d"),
    xfail("nn.functional.fractional_max_pool3d"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("nn.functional.glu"),
    xfail("nn.functional.grid_sample"),
    xfail("nn.functional.group_norm"),
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.hardsigmoid"),
    xfail("nn.functional.hardswish"),
    xfail("nn.functional.hardtanh"),
    xfail("nn.functional.huber_loss"),
    xfail("nn.functional.instance_norm"),
    xfail("nn.functional.interpolate", "area"),
    xfail("nn.functional.interpolate", "bicubic"),
    xfail("nn.functional.interpolate", "bilinear"),
    xfail("nn.functional.interpolate", "linear"),
    xfail("nn.functional.interpolate", "nearest"),
    xfail("nn.functional.interpolate", "trilinear"),
    xfail("nn.functional.kl_div"),
    xfail("nn.functional.l1_loss"),
    xfail("nn.functional.layer_norm"),
    xfail("nn.functional.leaky_relu"),
    xfail("nn.functional.linear"),
    xfail("nn.functional.local_response_norm"),
    xfail("nn.functional.logsigmoid"),
    xfail("nn.functional.margin_ranking_loss"),
    xfail("nn.functional.max_pool1d"),
    xfail("nn.functional.max_pool2d"),
    xfail("nn.functional.max_pool3d"),
    xfail("nn.functional.max_unpool1d"),
    xfail("nn.functional.max_unpool1d", "grad"),
    xfail("nn.functional.max_unpool2d"),
    xfail("nn.functional.max_unpool2d", "grad"),
    xfail("nn.functional.max_unpool3d"),
    xfail("nn.functional.max_unpool3d", "grad"),
    xfail("nn.functional.mish"),
    xfail("nn.functional.mse_loss"),
    xfail("nn.functional.multi_margin_loss"),
    xfail("nn.functional.multilabel_margin_loss"),
    xfail("nn.functional.multilabel_soft_margin_loss"),
    xfail("nn.functional.nll_loss"),
    xfail("nn.functional.normalize"),
    xfail("nn.functional.pad", "circular"),
    xfail("nn.functional.pad", "constant"),
    xfail("nn.functional.pad", "reflect"),
    xfail("nn.functional.pad", "replicate"),
    xfail("nn.functional.pairwise_distance"),
    xfail("nn.functional.pdist"),
    xfail("nn.functional.pixel_shuffle"),
    xfail("nn.functional.pixel_unshuffle"),
    xfail("nn.functional.poisson_nll_loss"),
    xfail("nn.functional.prelu"),
    xfail("nn.functional.relu6"),
    xfail("nn.functional.rrelu"),
    xfail("nn.functional.selu"),
    xfail("nn.functional.silu"),
    xfail("nn.functional.smooth_l1_loss"),
    xfail("nn.functional.soft_margin_loss"),
    xfail("nn.functional.softplus"),
    xfail("nn.functional.softshrink"),
    xfail("nn.functional.softsign"),
    xfail("nn.functional.threshold"),
    xfail("nn.functional.triplet_margin_loss"),
    xfail("nn.functional.triplet_margin_with_distance_loss"),
    xfail("nn.functional.unfold"),
    xfail("nn.functional.upsample_bilinear"),
    xfail("nn.functional.upsample_nearest"),
    xfail("nonzero"),
    xfail("norm"),
    xfail("norm", "fro"),
    xfail("norm", "inf"),
    xfail("norm", "nuc"),
    xfail("normal"),
    xfail("normal", "number_mean"),
    xfail("ormqr"),
    xfail("outer"),
    xfail("pca_lowrank"),
    xfail("pinverse"),
    xfail("polar"),
    xfail("polygamma", "polygamma_n_0"),
    xfail("polygamma", "polygamma_n_1"),
    xfail("polygamma", "polygamma_n_2"),
    xfail("polygamma", "polygamma_n_3"),
    xfail("polygamma", "polygamma_n_4"),
    xfail("pow"),
    xfail("put"),
    xfail("qr"),
    xfail("quantile"),
    xfail("rad2deg"),
    xfail("rand_like"),
    xfail("randint_like"),
    xfail("randn_like"),
    xfail("ravel"),
    xfail("reciprocal"),
    xfail("remainder"),
    xfail("renorm"),
    xfail("repeat_interleave"),
    xfail("reshape_as"),
    xfail("reshape"),
    xfail("resize_"),
    xfail("resize_as_"),
    xfail("roll"),
    xfail("rot90"),
    xfail("round"),
    xfail("round", "decimals_0"),
    xfail("round", "decimals_3"),
    xfail("round", "decimals_neg_3"),
    xfail("rsqrt"),
    xfail("rsub"),
    xfail("scatter_add"),
    xfail("scatter"),
    xfail("scatter_reduce", "amax"),
    xfail("scatter_reduce", "amin"),
    xfail("scatter_reduce", "mean"),
    xfail("scatter_reduce", "prod"),
    xfail("scatter_reduce", "sum"),
    xfail("searchsorted"),
    xfail("select"),
    xfail("select_scatter"),
    xfail("sign"),
    xfail("signbit"),
    xfail("sin"),
    xfail("sinc"),
    xfail("sinh"),
    xfail("slice_scatter"),
    xfail("sort"),
    xfail("sparse.sampled_addmm"),
    xfail("special.airy_ai"),
    xfail("special.bessel_j0"),
    xfail("special.bessel_j1"),
    xfail("special.bessel_y0"),
    xfail("special.bessel_y1"),
    xfail("special.chebyshev_polynomial_t"),
    xfail("special.chebyshev_polynomial_u"),
    xfail("special.entr"),
    xfail("special.erfcx"),
    xfail("special.hermite_polynomial_h"),
    xfail("special.hermite_polynomial_he"),
    xfail("special.i0e"),
    xfail("special.i1"),
    xfail("special.i1e"),
    xfail("special.laguerre_polynomial_l"),
    xfail("special.log_ndtr"),
    xfail("special.modified_bessel_i0"),
    xfail("special.modified_bessel_i1"),
    xfail("special.modified_bessel_k0"),
    xfail("special.modified_bessel_k1"),
    xfail("special.ndtr"),
    xfail("special.ndtri"),
    xfail("special.polygamma", "special_polygamma_n_0"),
    xfail("special.scaled_modified_bessel_k0"),
    xfail("special.scaled_modified_bessel_k1"),
    xfail("special.spherical_bessel_j0"),
    xfail("special.xlog1py"),
    xfail("special.zeta"),
    xfail("split"),
    xfail("split", "list_args"),
    xfail("split_with_sizes"),
    xfail("sqrt"),
    xfail("square"),
    xfail("squeeze"),
    xfail("stack"),
    xfail("std"),
    xfail("std_mean"),
    xfail("stft"),
    xfail("sub"),
    xfail("sum_to_size"),
    xfail("svd"),
    xfail("svd_lowrank"),
    xfail("symeig"),
    xfail("t"),
    xfail("take_along_dim"),
    xfail("take"),
    xfail("tan"),
    xfail("tensor_split"),
    xfail("tensordot"),
    xfail("to_sparse"),
    xfail("topk"),
    xfail("trace"),
    xfail("trapezoid"),
    xfail("trapz"),
    xfail("triangular_solve"),
    xfail("tril"),
    xfail("triu"),
    xfail("true_divide"),
    xfail("trunc"),
    xfail("unbind"),
    xfail("unfold"),
    xfail("unflatten"),
    xfail("unique_consecutive"),
    xfail("unique"),
    xfail("var"),
    xfail("var_mean"),
    xfail("vdot"),
    xfail("view_as_complex"),
    xfail("view_as"),
    xfail("view"),  # view related op only works with certain sharding dims
    xfail("vsplit"),
    xfail("vstack"),
    xfail("where"),
    xfail("xlogy"),
    xfail("zero_"),
    xfail("zeros_like"),
    # ops inside this might even fail without dtensor
    # tests, as we rescale op db common test size factor (i.e. L, M, S)
    # which triggered the orignal function run failures with input
    # generation becomes wrong, we skip them for now but should enable later.
    # TODO: need to clean this list and remove all cases
    skip("argwhere"),
    skip("cumprod"),
    skip("__rmatmul__"),
    skip("softmax"),
    skip("meshgrid", "list_of_tensors"),
    skip("meshgrid", "variadic_tensors"),
    skip("nn.functional.softmin"),
    skip("nn.functional.embedding"),
    skip("nn.functional.embedding_bag"),
    skip("nn.functional.feature_alpha_dropout", "with_train"),
    skip("nn.functional.feature_alpha_dropout", "without_train"),
    skip("nn.functional.hinge_embedding_loss"),
    skip("nn.functional.cosine_embedding_loss"),
    skip("fft.hfft"),
    skip("fft.hfft2"),
    skip("fft.hfft2"),
    skip("fft.hfftn"),
    skip("fft.ifftn"),
    skip("fft.irfft"),
    skip("istft"),
    skip("isclose"),
    skip("isreal"),
    skip("matmul"),
    skip("_masked.mean"),
    skip("_masked.var"),
    skip("_masked.std"),
    skip("_masked.normalize"),
    skip("ones_like"),
    skip("prod"),
    skip("segment_reduce", "lengths"),
    skip("segment_reduce", "offsets"),
}


# Add a list of ops that are currently failing BW pass
skip_bw = [
    None,  # corresponds to the transpose ops 'H' and 'T'
    "torch.isfinite",
    "torch.eq",
    "torch.isnan",
    "torch.conj_physical",
]


def run_dtensor_crossref(test_case, func, args, kwargs):
    to_dtensor = DTensorConverter(test_case.mesh, args, kwargs)

    # TODO: also handle cases where func raise an exception
    rs = func(*args, **kwargs)

    def to_replicate(e: object) -> object:
        return e.redistribute(
                test_case.mesh, test_case.mesh.ndim * [Replicate()]
            ) if isinstance(e, DTensor) else e

    try:
        # Suppress warnings, this doesn't matter for test_meta.py
        # but it does matter if you want to use this decorator
        # for cross-ref testing, as some tests may be looking at
        # errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # for every comb of sharding choices, we test if it works
            for dtensor_args, dtensor_kwargs in to_dtensor:
                # Only attempt if we managed to convert all tensors to DTensor
                # (if any of them failed, we're in a mixed tensor situation and
                # this is not allowed in DTensor)
                if to_dtensor.successful():
                    # Handle special cases first if there's any
                    # Suppress warnings, this doesn't matter for test_meta.py
                    # but it does matter if you want to use this decorator
                    # for cross-ref testing, as some tests may be looking at
                    # errors
                    dtensor_rs = func(*dtensor_args, **dtensor_kwargs)

                    # redistribute/all_gather the results to compare with normal output
                    dtensor_rs = tree_map(to_replicate, dtensor_rs)
                    try:
                        if resolve_name(func) not in skip_bw:
                            if isinstance(dtensor_rs, DTensor):
                                dtensor_rs.to_local().sum().backward()
                            elif isinstance(dtensor_rs, tuple):
                                dtensor_rs[0].to_local().sum().backward()

                    except Exception as e:
                        # TODO(anj): Remove this guard exception after gaining more confidence.
                        if torch.distributed.get_rank() == 0:
                            print(
                                f"failed to run BW: {resolve_name(func)}, {func}, {str(e)})"
                            )
                    assert_ref_dtensor_equal(test_case, dtensor_rs, rs)
                else:
                    raise RuntimeError(
                        f"failed to convert args to DTensor; "
                        f"originally (*{args}, **{kwargs})"
                    )
    except Exception as e:
        raise RuntimeError(
            f"failed to run: {resolve_name(func)}, with (*{args}, **{kwargs})"
        ) from e

    return rs


def check_dtensor_func(test_case, test_func, opinfo, dry_run=False):
    try:
        test_func()
    except Exception:
        test_case.destroy_pg()
        if not dry_run:
            raise
        if dist.get_rank() == 0:
            if opinfo.variant_test_name:
                print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
            else:
                print(f"xfail('{opinfo.name}'),")
    else:
        test_case.destroy_pg()


class TestDTensorOps(DistTensorTestBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    # only allow float dytpe for now, we can relax this constraint
    # when feel necessary later (i.e when adding quantization support).
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @suppress_warnings
    @ops(dtensor_lagging_op_db, allowed_dtypes=(torch.float,))
    @skipOps("TestDTensorOps", "test_dtensor_op_db", dtensor_fails)
    def test_dtensor_op_db(self, dtype, op):
        pg_backend = "nccl" if DEVICE_TYPE == "cuda" else "gloo"
        if pg_backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.init_pg(backend=pg_backend)
        self.mesh = DeviceMesh(DEVICE_TYPE, torch.arange(self.world_size))

        # test each op with dist tensor inputs and normal inputs
        def test():
            samples = op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=True)
            for sample_input in samples:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs

                run_dtensor_crossref(self, op.op, args, kwargs)
                # we need to figure out a way to test the out variant, out variant testing
                # is tricky, as we need to pre allocate the dtensor out, some of them rely
                # on sharding placements to be pre-known (i.e. mm.out)
                # if isinstance(expected, torch.Tensor) and op.supports_out:
                #     func(*args, **kwargs, out=expected)

        check_dtensor_func(self, test, op)


# only instantiate tests for DEVICE_TYPE alone (i.e. either CPU or GPU)
instantiate_device_type_tests(TestDTensorOps, globals(), only_for=(DEVICE_TYPE))


if __name__ == "__main__":
    run_tests()
