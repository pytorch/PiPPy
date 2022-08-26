# Owner(s): ["module: distributed"]

import torch
import os
from enum import Enum
import sys
import atexit
import re
import unittest
import warnings
from typing import Dict, Set, List

from torch.overrides import resolve_name
from torch.utils._pytree import tree_map, tree_flatten
import torch.utils._python_dispatch
from torch.testing._internal.common_utils import (
    TestCase,
    skipIfCrossRef,
    suppress_warnings,
    TEST_WITH_ASAN,
    run_tests,
)
from torch.testing._internal.common_device_type import (
    ops,
    instantiate_device_type_tests,
)
import torch.testing._internal.common_methods_invocations as common_ops
from torchgen.model import OperatorName

from spmd import DeviceMesh, Replicate
from spmd.test._utils import DistTensorTestBase, TEST_SKIPS, DTensorConverter


bf16 = torch.bfloat16
f64 = torch.float64
f32 = torch.float32
f16 = torch.float16
c32 = torch.complex32
c64 = torch.complex64
c128 = torch.complex128
i8 = torch.int8
i16 = torch.int16
i32 = torch.int32
i64 = torch.int64
b8 = torch.bool
u8 = torch.uint8

dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

# rewrite common size variables to sth can be sharded evenly
# we can enable uneven shards later, but need to adjust more on
# sample inputs (i.e. view/reshape need to adjust shape size as well)
common_ops.L = 24
common_ops.M = 12
common_ops.S = 4
common_ops.XS = 2

# override debug strict in this file to allow easier testing, we will remove
# debug assert in next few PRs
import spmd.tensor.dispatch as dtensor_dispatch

dtensor_dispatch._DEBUG_STRICT = True


def assert_ref_dtensor_equal(test_case, dtensor_rs, rs, msg_callable):
    mesh = test_case.mesh
    flat_dtensor_rs, _ = tree_flatten(dtensor_rs)
    flat_rs, _ = tree_flatten(rs)
    test_case.assertEqual(len(flat_dtensor_rs), len(flat_rs))
    for i, dtensor_r, r in zip(range(len(flat_rs)), flat_dtensor_rs, flat_rs):

        def test_assert(cond, msg):
            if not cond:
                raise RuntimeError(f"output {i}: {msg_callable(msg)}")

        if not isinstance(r, torch.Tensor):
            continue
        test_assert(
            isinstance(dtensor_r, torch.Tensor),
            f"but real {i}th result is Tensor",
        )
        test_assert(dtensor_r.dtype == r.dtype, f"but real dtype was {r.dtype}")
        test_assert(dtensor_r.shape == r.shape, f"but real shape was {r.shape}")
        # NOTE: stride checking is currently disabled
        # See https://github.com/pytorch/pytorch/issues/78050
        # same_strides, _ = prims.utils.check_significant_strides(meta_r, r)
        # test_assert(same_strides, f"but real stride was {r.stride()}")
        test_assert(
            dtensor_r.requires_grad == r.requires_grad,
            f"but real requires_grad was {r.requires_grad}",
        )

        # redistribute/all_gather the results to compare with normal output
        full_out = dtensor_r.redistribute(
            mesh, mesh.ndim * [Replicate()]
        ).to_local()
        test_case.assertEqual(full_out, r)


# This environment variable controls whether or not we print expected failure
# lists at the end of a test suite run.  The intended usage looks like this:
#
# 1. Run `PYTORCH_COLLECT_EXPECT=1 python test/test_dtensor_ops.py` on a CUDA build
#    of PyTorch that has LAPACK/MAGMA installed.
# 2. Given the printed skip/xfail list, add them to the corresponding lists;
#    torch.* entries go in meta_function and aten.* entries go in meta_dispatch.
#    If there are preexisting entries, you need to merge in the entries.
#
# This is somewhat manual but typically you shouldn't need to do this, unless
# you've made a major change (e.g., added a new dtype to PyTorch) and need to
# refresh the lists.  If you want to do it from scratch, just clear out the
# preexisting lists before running.
#
# WARNING: Python dict literals will silently ignore duplicate keys
COLLECT_EXPECT = os.getenv("PYTORCH_COLLECT_EXPECT", "0") == "1"

seen_succeeded: Dict[torch._ops.OpOverload, Set[torch.dtype]] = {}
seen_failed: Dict[torch._ops.OpOverload, Set[torch.dtype]] = {}
failed_reasons: Dict[torch._ops.OpOverload, List[str]] = {}
dispatch_functions: Dict[torch._ops.OpOverload, Set[torch.dtype]] = {}


def print_seen():
    expected_failures = []
    skips = []

    for op, failed_dtypes in dispatch_functions.items():
        expected_failures.append(
            f"    {op}: {{{dtype_abbrs[failed_dtypes.pop()]}}},"
        )

    def fmt_dtypes(dtypes):
        r = ", ".join(sorted(dtype_abbrs[d] for d in dtypes))
        return "{" + r + "}"

    for op, failed_dtypes in seen_failed.items():
        ops = resolve_name(op)
        succeeded_dtypes = seen_succeeded.get(op, set())
        expected_failures_dtypes = failed_dtypes - succeeded_dtypes
        skips_dtypes = failed_dtypes & succeeded_dtypes
        reasons = ""
        if failed_reasons[op]:
            reasons = "  # " + ", ".join(sorted(failed_reasons[op]))
        if expected_failures_dtypes:
            expected_failures.append(
                f"    {ops}: {fmt_dtypes(expected_failures_dtypes)},{reasons}"
            )
        if skips_dtypes:
            skips.append(f"    {ops}: {fmt_dtypes(skips_dtypes)},")
    expected_failures.sort()
    skips.sort()
    nl = "\n"
    print(
        f"""\
expected_failures = {{
{nl.join(expected_failures)}
}}

skips = {{
{nl.join(skips)}
}}
"""
    )


if COLLECT_EXPECT:
    atexit.register(print_seen)

# Success forces pass; failure forces fail; skip unconditionally skips testing
ExpectTestState = Enum("ExpectTestState", ("SUCCESS", "XFAILURE", "SKIP"))

# unlike print produce strides
def verbose_print(e):
    class Lit:
        def __init__(self, s):
            self.s = s

        def __repr__(self):
            return self.s

    def go(t):
        if isinstance(t, torch.Tensor):
            return Lit(f"{t} stride={t.stride()}")
        else:
            return t

    return repr(tree_map(go, e))


def run_dtensor_crossref(
    test_case,
    test_expect,
    func,
    args,
    kwargs,
    *,
    dtype,
    device_type,
):
    do_dtensor = test_expect is not ExpectTestState.SKIP
    to_dtensor = DTensorConverter(test_case.mesh, args, kwargs)

    # TODO: also handle cases where func raise an exception
    rs = func(*args, **kwargs)

    if do_dtensor:
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
                        delim = ",\n  "
                        assert_ref_dtensor_equal(
                            test_case,
                            dtensor_rs,
                            rs,
                            lambda msg: f"""\
meta disagrees with real impl:
{resolve_name(func)}(
{delim.join(map(verbose_print, dtensor_args))},
{delim.join(k + ": " + verbose_print(v) for k, v in dtensor_kwargs.items())}
) = (
{verbose_print(dtensor_rs)}
)
{msg}
""",
                        )
                    else:
                        raise RuntimeError(
                            f"failed to convert args to DTensor; "
                            f"originally (*{args}, **{kwargs})"
                        )
        except Exception as e:
            if test_expect is ExpectTestState.XFAILURE:
                return rs
            seen_failed.setdefault(func, set()).add(dtype)
            if isinstance(e, NotImplementedError):
                m = RE_NOT_IMPLEMENTED_MSG.search(e.args[0])
                if m:
                    failed_reasons[func].add(m.group(1))
            if COLLECT_EXPECT:
                return rs
            raise RuntimeError(
                f"""\
failed to run: {resolve_name(func)}(
*{verbose_print(dtensor_args)},
**{verbose_print(dtensor_kwargs)}
)"""
            ) from e
        else:
            seen_succeeded.setdefault(func, set()).add(dtype)
            if test_expect is ExpectTestState.XFAILURE and not COLLECT_EXPECT:
                raise RuntimeError(f"unexpected success {resolve_name(func)}")

    return rs


RE_NOT_IMPLEMENTED_MSG = re.compile(r"Could not run '([^']+)' with arguments ")


"""
# This is some sample code for how we could dump these dicts into YAML
# file for easier reading/writing
import yaml
print(yaml.dump(
  {resolve_name(k): [dtype_abbrs[d] for d in v]
   for k, v in meta_function_expected_failures.items()}, default_flow_style=None))
import sys
sys.exit()
"""

aten = torch.ops.aten

# these always fail
dtensor_dispatch_expected_failures: Dict[
    torch._ops.OpOverload, Set[torch.dtype]
] = {
    aten._adaptive_avg_pool2d.default: {f32},
    aten._adaptive_avg_pool3d.default: {f32},
    aten._cdist_forward.default: {f32},
    aten._conj.default: {f32},
    aten._conj_physical.default: {f32},
    aten._convolution.default: {f32},
    aten._ctc_loss.default: {f32},
    aten._embedding_bag_forward_only.default: {f32},
    aten._euclidean_dist.default: {f32},
    aten._fft_c2c.default: {f32},
    aten._fft_c2r.default: {f32},
    aten._fft_r2c.default: {f32},
    aten._histogramdd_bin_edges.default: {f32},
    aten._histogramdd_from_bin_cts.default: {f32},
    aten._histogramdd_from_bin_tensors.default: {f32},
    aten._linalg_check_errors.default: {f32},
    aten._linalg_det.default: {f32},
    aten._linalg_eigh.default: {f32},
    aten._linalg_slogdet.default: {f32},
    aten._linalg_solve_ex.default: {f32},
    aten._linalg_svd.default: {f32},
    aten._local_scalar_dense.default: {f32},
    aten._log_softmax.default: {f32},
    aten._pdist_forward.default: {f32},
    aten._reshape_alias.default: {f32},
    aten._softmax.default: {f32},
    aten._to_copy.default: {f32},
    aten._trilinear.default: {f32},
    aten._unique2.default: {f32},
    aten._unsafe_view.default: {f32},
    aten.abs.default: {f32},
    aten.acos.default: {f32},
    aten.acosh.default: {f32},
    aten.adaptive_max_pool2d.default: {f32},
    aten.adaptive_max_pool3d.default: {f32},
    aten.add.Scalar: {f32},
    aten.add.Tensor: {f32},
    aten.add_.Tensor: {f32},
    aten.addbmm.default: {f32},
    aten.addcdiv.default: {f32},
    aten.addcmul.default: {f32},
    aten.addmv.default: {f32},
    aten.addr.default: {f32},
    aten.alias.default: {f32},
    aten.all.default: {f32},
    aten.all.dim: {f32},
    aten.allclose.default: {f32},
    aten.amax.default: {f32},
    aten.amin.default: {f32},
    aten.aminmax.default: {f32},
    aten.angle.default: {f32},
    aten.any.default: {f32},
    aten.any.dim: {f32},
    aten.arange.default: {f32},
    aten.arange.start: {f32},
    aten.arange.start_step: {f32},
    aten.argmax.default: {f32},
    aten.argmin.default: {f32},
    aten.argwhere.default: {f32},
    aten.as_strided.default: {f32},
    aten.as_strided_scatter.default: {f32},
    aten.asin.default: {f32},
    aten.asinh.default: {f32},
    aten.atan.default: {f32},
    aten.atan2.default: {f32},
    aten.atanh.default: {f32},
    aten.avg_pool2d.default: {f32},
    aten.avg_pool3d.default: {f32},
    aten.baddbmm.default: {f32},
    aten.bernoulli.default: {f32},
    aten.bernoulli_.float: {f32},
    aten.binary_cross_entropy.default: {f32},
    aten.binary_cross_entropy_with_logits.default: {f32},
    aten.bitwise_and_.Tensor: {f32},
    aten.bitwise_or_.Tensor: {f32},
    aten.block_diag.default: {f32},
    aten.bmm.default: {f32},
    aten.bucketize.Tensor: {f32},
    aten.ceil.default: {f32},
    aten.ceil_.default: {f32},
    aten.celu.default: {f32},
    aten.cholesky.default: {f32},
    aten.cholesky_inverse.default: {f32},
    aten.cholesky_solve.default: {f32},
    aten.clamp.Tensor: {f32},
    aten.clamp.default: {f32},
    aten.clamp_.default: {f32},
    aten.clamp_max.Tensor: {f32},
    aten.clamp_min.Tensor: {f32},
    aten.clamp_min.default: {f32},
    aten.clamp_min_.default: {f32},
    aten.col2im.default: {f32},
    aten.complex.default: {f32},
    aten.complex.out: {f32},
    aten.constant_pad_nd.default: {f32},
    aten.convolution.default: {f32},
    aten.copy_.default: {f32},
    aten.copysign.Tensor: {f32},
    aten.cos.default: {f32},
    aten.cosh.default: {f32},
    aten.count_nonzero.default: {f32},
    aten.count_nonzero.dim_IntList: {f32},
    aten.cummax.default: {f32},
    aten.cummin.default: {f32},
    aten.cumprod.default: {f32},
    aten.cumsum.default: {f32},
    aten.deg2rad.default: {f32},
    aten.diag.default: {f32},
    aten.diag_embed.default: {f32},
    aten.diagonal.default: {f32},
    aten.diagonal_scatter.default: {f32},
    aten.digamma.default: {f32},
    aten.dist.default: {f32},
    aten.div.Scalar: {f32},
    aten.div.Tensor: {f32},
    aten.div.Tensor_mode: {f32},
    aten.div_.Scalar: {f32},
    aten.dot.default: {f32},
    aten.eig.default: {f32},
    aten.elu.default: {f32},
    aten.embedding.default: {f32},
    aten.embedding_renorm_.default: {f32},
    aten.empty.memory_format: {f32},
    aten.empty_like.default: {f32},
    aten.eye.m: {f32},
    aten.eq.Scalar: {f32},
    aten.eq.Tensor: {f32},
    aten.equal.default: {f32},
    aten.erf.default: {f32},
    aten.erfc.default: {f32},
    aten.erfinv.default: {f32},
    aten.exp.default: {f32},
    aten.exp2.default: {f32},
    aten.expm1.default: {f32},
    aten.eye.default: {f32},
    aten.fill_.Scalar: {f32},
    aten.flip.default: {f32},
    aten.floor.default: {f32},
    aten.floor_divide.default: {f32},
    aten.fmax.default: {f32},
    aten.fmin.default: {f32},
    aten.fmod.Tensor: {f32},
    aten.frac.default: {f32},
    aten.fractional_max_pool2d.default: {f32},
    aten.fractional_max_pool3d.default: {f32},
    aten.frexp.Tensor: {f32},
    aten.full.default: {f32},
    aten.full_like.default: {f32},
    aten.gather.default: {f32},
    aten.ge.Scalar: {f32},
    aten.ge.Tensor: {f32},
    aten.geqrf.default: {f32},
    aten.glu.default: {f32},
    aten.grid_sampler_2d.default: {f32},
    aten.grid_sampler_3d.default: {f32},
    aten.gt.Tensor: {f32},
    aten.hardshrink.default: {f32},
    aten.hardsigmoid.default: {f32},
    aten.hardswish.default: {f32},
    aten.hardtanh.default: {f32},
    aten.heaviside.default: {f32},
    aten.histc.default: {f32},
    aten.histogram.bin_ct: {f32},
    aten.histogram.bins_tensor: {f32},
    aten.huber_loss.default: {f32},
    aten.hypot.default: {f32},
    aten.i0.default: {f32},
    aten.igamma.default: {f32},
    aten.igammac.default: {f32},
    aten.im2col.default: {f32},
    aten.index.Tensor: {f32},
    aten.index_add.default: {f32},
    aten.index_copy.default: {f32},
    aten.index_fill.int_Scalar: {f32},
    aten.index_put.default: {f32},
    aten.index_reduce.default: {f32},
    aten.index_select.default: {f32},
    aten.inverse.default: {f32},
    aten.isin.Tensor_Tensor: {f32},
    aten.isinf.default: {f32},
    aten.isnan.default: {f32},
    aten.isneginf.default: {f32},
    aten.isposinf.default: {f32},
    aten.kthvalue.default: {f32},
    aten.le.Scalar: {f32},
    aten.le.Tensor: {f32},
    aten.leaky_relu.default: {f32},
    aten.lerp.Scalar: {f32},
    aten.lerp.Tensor: {f32},
    aten.lerp_.Tensor: {f32},
    aten.lgamma.default: {f32},
    aten.lift_fresh.default: {f32},
    aten.linalg_cholesky_ex.default: {f32},
    aten.linalg_cross.default: {f32},
    aten.linalg_eig.default: {f32},
    aten.linalg_householder_product.default: {f32},
    aten.linalg_inv_ex.default: {f32},
    aten.linalg_ldl_factor_ex.default: {f32},
    aten.linalg_ldl_solve.default: {f32},
    aten.linalg_lstsq.default: {f32},
    aten.linalg_lu.default: {f32},
    aten.linalg_lu_factor_ex.default: {f32},
    aten.linalg_lu_solve.default: {f32},
    aten.linalg_matrix_exp.default: {f32},
    aten.linalg_pinv.atol_rtol_tensor: {f32},
    aten.linalg_qr.default: {f32},
    aten.linalg_solve_triangular.default: {f32},
    aten.linalg_vector_norm.default: {f32},
    aten.linspace.default: {f32},
    aten.log.default: {f32},
    aten.log10.default: {f32},
    aten.log1p.default: {f32},
    aten.log2.default: {f32},
    aten.log_sigmoid_forward.default: {f32},
    aten.logaddexp.default: {f32},
    aten.logaddexp2.default: {f32},
    aten.logcumsumexp.default: {f32},
    aten.logical_and.default: {f32},
    aten.logical_and_.default: {f32},
    aten.logical_not.default: {f32},
    aten.logical_not_.default: {f32},
    aten.logical_or.default: {f32},
    aten.logical_xor.default: {f32},
    aten.logit.default: {f32},
    aten.logsumexp.default: {f32},
    aten.logspace.default: {f32},
    aten.lt.Scalar: {f32},
    aten.lt.Tensor: {f32},
    aten.lu_unpack.default: {f32},
    aten.masked_fill.Scalar: {f32},
    aten.masked_fill.Tensor: {f32},
    aten.masked_fill_.Scalar: {f32},
    aten.masked_scatter.default: {f32},
    aten.masked_select.default: {f32},
    aten.max.default: {f32},
    aten.max.dim: {f32},
    aten.max_pool2d_with_indices.default: {f32},
    aten.max_pool3d_with_indices.default: {f32},
    aten.max_unpool2d.default: {f32},
    aten.max_unpool3d.default: {f32},
    aten.maximum.default: {f32},
    aten.mean.default: {f32},
    aten.mean.dim: {f32},
    aten.median.default: {f32},
    aten.median.dim: {f32},
    aten.min.default: {f32},
    aten.min.dim: {f32},
    aten.minimum.default: {f32},
    aten.mish.default: {f32},
    aten.mm.default: {f32},
    aten.mode.default: {f32},
    aten.mse_loss.default: {f32},
    aten.mul.Scalar: {f32},
    aten.mul.Tensor: {f32},
    aten.mul_.Tensor: {f32},
    aten.multi_margin_loss.default: {f32},
    aten.multilabel_margin_loss_forward.default: {f32},
    aten.multinomial.default: {f32},
    aten.mv.default: {f32},
    aten.mvlgamma.default: {f32},
    aten.nan_to_num.default: {f32},
    aten.nanmedian.default: {f32},
    aten.nanmedian.dim: {f32},
    aten.nansum.default: {f32},
    aten.native_batch_norm.default: {f32},
    aten.native_group_norm.default: {f32},
    aten.native_layer_norm.default: {f32},
    aten.ne.Scalar: {f32},
    aten.ne.Tensor: {f32},
    aten.neg.default: {f32},
    aten.new_empty.default: {f32},
    aten.new_zeros.default: {f32},
    aten.new_full.default: {f32},
    aten.new_ones.default: {f32},
    aten.nextafter.default: {f32},
    aten.nll_loss2d_forward.default: {f32},
    aten.nll_loss_forward.default: {f32},
    aten.nonzero.default: {f32},
    aten.norm.ScalarOpt_dim: {f32},
    aten.normal.Tensor_Tensor: {f32},
    aten.normal.Tensor_float: {f32},
    aten.normal.float_Tensor: {f32},
    aten.ones.default: {f32},
    aten.ormqr.default: {f32},
    aten.pixel_shuffle.default: {f32},
    aten.pixel_unshuffle.default: {f32},
    aten.polar.default: {f32},
    aten.polygamma.default: {f32},
    aten.pow.Scalar: {f32},
    aten.pow.Tensor_Scalar: {f32},
    aten.pow.Tensor_Tensor: {f32},
    aten.prelu.default: {f32},
    aten.prod.default: {f32},
    aten.prod.dim_int: {f32},
    aten.put.default: {f32},
    aten.put_.default: {f32},
    aten.rad2deg.default: {f32},
    aten.rand.default: {f32},
    aten.randint_like.default: {f32},
    aten.randint_like.low_dtype: {f32},
    aten.randn.default: {f32},
    aten.randn_like.default: {f32},
    aten.reciprocal.default: {f32},
    aten.reflection_pad1d.default: {f32},
    aten.reflection_pad2d.default: {f32},
    aten.reflection_pad3d.default: {f32},
    aten.remainder.Tensor: {f32},
    aten.renorm.default: {f32},
    aten.repeat.default: {f32},
    aten.repeat_interleave.Tensor: {f32},
    aten.replication_pad1d.default: {f32},
    aten.replication_pad2d.default: {f32},
    aten.replication_pad3d.default: {f32},
    aten.resize_.default: {f32},
    aten.resize_as_.default: {f32},
    aten.roll.default: {f32},
    aten.rot90.default: {f32},
    aten.round.decimals: {f32},
    aten.round.default: {f32},
    aten.rrelu_with_noise.default: {f32},
    aten.rsqrt.default: {f32},
    aten.rsub.Scalar: {f32},
    aten.rsub.Tensor: {f32},
    aten.scalar_tensor.default: {f32},
    aten.scatter.reduce: {f32},
    aten.scatter.src: {f32},
    aten.scatter.value: {f32},
    aten.scatter.value_reduce: {f32},
    aten.scatter_add.default: {f32},
    aten.scatter_reduce.two: {f32},
    aten.searchsorted.Tensor: {f32},
    aten.segment_reduce.default: {f32},
    aten.select.int: {f32},
    aten.select_scatter.default: {f32},
    aten.sgn.default: {f32},
    aten.sign.default: {f32},
    aten.signbit.default: {f32},
    aten.silu.default: {f32},
    aten.sin.default: {f32},
    aten.sinc.default: {f32},
    aten.sinh.default: {f32},
    aten.slice.Tensor: {f32},
    aten.slice_scatter.default: {f32},
    aten.smooth_l1_loss.default: {f32},
    aten.soft_margin_loss.default: {f32},
    aten.softplus.default: {f32},
    aten.softshrink.default: {f32},
    aten.sort.default: {f32},
    aten.sort.stable: {f32},
    aten.sparse_sampled_addmm.default: {f32},
    aten.special_airy_ai.default: {f32},
    aten.special_bessel_j0.default: {f32},
    aten.special_bessel_j1.default: {f32},
    aten.special_bessel_y0.default: {f32},
    aten.special_bessel_y1.default: {f32},
    aten.special_chebyshev_polynomial_t.default: {f32},
    aten.special_chebyshev_polynomial_u.default: {f32},
    aten.special_entr.default: {f32},
    aten.special_erfcx.default: {f32},
    aten.special_hermite_polynomial_h.default: {f32},
    aten.special_hermite_polynomial_he.default: {f32},
    aten.special_i0e.default: {f32},
    aten.special_i1.default: {f32},
    aten.special_i1e.default: {f32},
    aten.special_laguerre_polynomial_l.default: {f32},
    aten.special_log_ndtr.default: {f32},
    aten.special_modified_bessel_i0.default: {f32},
    aten.special_modified_bessel_i1.default: {f32},
    aten.special_modified_bessel_k0.default: {f32},
    aten.special_modified_bessel_k1.default: {f32},
    aten.special_ndtri.default: {f32},
    aten.special_scaled_modified_bessel_k0.default: {f32},
    aten.special_scaled_modified_bessel_k1.default: {f32},
    aten.special_spherical_bessel_j0.default: {f32},
    aten.special_xlog1py.default: {f32},
    aten.special_zeta.default: {f32},
    aten.split.Tensor: {f32},
    aten.split_with_sizes.default: {f32},
    aten.sqrt.default: {f32},
    aten.sqrt_.default: {f32},
    aten.squeeze.default: {f32},
    aten.squeeze.dim: {f32},
    aten.squeeze_.dim: {f32},
    aten.stack.default: {f32},
    aten.std.correction: {f32},
    aten.std_mean.correction: {f32},
    aten.sub.Scalar: {f32},
    aten.sub.Tensor: {f32},
    aten.sub_.Tensor: {f32},
    aten.sum.IntList_out: {f32},
    aten.symeig.default: {f32},
    aten.take.default: {f32},
    aten.tan.default: {f32},
    aten.tanh.default: {f32},
    aten.threshold.default: {f32},
    aten.threshold_backward.default: {f32},
    aten.to_sparse.default: {f32},
    aten.to_sparse.sparse_dim: {f32},
    aten.topk.default: {f32},
    aten.trace.default: {f32},
    aten.transpose_.default: {f32},
    aten.triangular_solve.default: {f32},
    aten.tril.default: {f32},
    aten.triu.default: {f32},
    aten.trunc.default: {f32},
    aten.unbind.int: {f32},
    aten.unfold.default: {f32},
    aten.unique_consecutive.default: {f32},
    aten.unique_dim.default: {f32},
    aten.unsqueeze_.default: {f32},
    aten.upsample_bicubic2d.vec: {f32},
    aten.upsample_bilinear2d.vec: {f32},
    aten.upsample_linear1d.vec: {f32},
    aten.upsample_nearest1d.vec: {f32},
    aten.upsample_nearest2d.vec: {f32},
    aten.upsample_nearest3d.vec: {f32},
    aten.upsample_trilinear3d.vec: {f32},
    aten.var.correction: {f32},
    aten.var_mean.correction: {f32},
    aten.vdot.default: {f32},
    aten.view_as_complex.default: {f32},
    aten.view_as_real.default: {f32},
    aten.where.self: {f32},
    aten.xlogy.Tensor: {f32},
    aten.zero_.default: {f32},
    aten.zeros.default: {f32},
    aten.zeros_like.default: {f32},
}

# these sometimes pass and sometimes fail
# i.e. view only works with certain sharding dims
# we need to remove many of them from list once op
# get full support with varying sharding specs
dtensor_dispatch_skips: Dict[torch._ops.OpOverload, Set[torch.dtype]] = {
    aten.new_empty_strided.default: {f32},
    aten.view.default: {f32},
    aten.unsqueeze.default: {f32},
    aten.repeat.default: {f32},
    aten.cat.default: {f32},
    aten._softmax.default: {f32},
    aten.addmm.default: {f32},
    aten.mm.default: {f32},
    aten.bmm.default: {f32},
    aten.sum.default: {f32},
    aten.sum.dim_IntList: {f32},
    aten.t.default: {f32},
    aten.transpose.int: {f32},
}

dtensor_dispatch_device_expected_failures: Dict[
    str, Dict[torch._ops.OpOverload, Set[torch.dtype]]
] = {}
dtensor_dispatch_device_skips: Dict[
    str, Dict[torch._ops.OpOverload, Set[torch.dtype]]
] = {}

dtensor_dispatch_device_expected_failures["cpu"] = {}
dtensor_dispatch_device_expected_failures["cuda"] = {}
dtensor_dispatch_device_skips["cpu"] = {}
dtensor_dispatch_device_skips["cuda"] = {}

# ops inside this might even fail without dtensor
# tests, as we rescale op db common test size factor (i.e. L, M, S)
# which triggered the orignal function run failures with input
# generation becomes wrong, we skip them for now but should enable later.
# TODO: need to clean this list and remove all cases
op_inputs_skips = {
    "argwhere",
    "cumprod",
    "__rmatmul__",
    "softmax",
    "meshgrid",
    "nn.functional.softmin",
    "nn.functional.embedding",
    "nn.functional.embedding_bag",
    "nn.functional.feature_alpha_dropout",
    "nn.functional.hinge_embedding_loss",
    "nn.functional.cosine_embedding_loss",
    "fft.hfft",
    "fft.hfft2",
    "fft.hfft2",
    "fft.hfftn",
    "fft.ifftn",
    "fft.irfft",
    "istft",
    "isclose",
    "isreal",
    "matmul",
    "_masked.mean",
    "_masked.var",
    "_masked.std",
    "_masked.normalize",
    "ones_like",
    "prod",
    "segment_reduce",
}


class DTensorCrossRefDispatchMode(
    torch.utils._python_dispatch.TorchDispatchMode
):
    test_case: TestCase
    device: torch.device
    dtype: torch.dtype

    def __init__(self, test_case, *, device, dtype):
        self.test_case = test_case
        # save TLS
        self.precision = test_case.precision
        self.rel_tol = test_case.rel_tol
        self.device_type = torch.device(device).type
        self.dtype = dtype

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        dispatch_functions.setdefault(func, set()).add(self.dtype)
        kwargs = kwargs or {}

        self.test_case.precision = self.precision
        self.test_case.rel_tol = self.rel_tol

        if self.dtype in dtensor_dispatch_skips.get(func, set()):
            test_expect = ExpectTestState.SKIP
        elif self.dtype in dtensor_dispatch_device_skips[self.device_type].get(
            func, set()
        ):
            test_expect = ExpectTestState.SKIP
        elif self.dtype in dtensor_dispatch_expected_failures.get(func, set()):
            test_expect = ExpectTestState.XFAILURE
        elif self.dtype in dtensor_dispatch_device_expected_failures[
            self.device_type
        ].get(func, set()):
            test_expect = ExpectTestState.XFAILURE
        else:
            test_expect = ExpectTestState.SUCCESS

        return run_dtensor_crossref(
            self.test_case,
            test_expect,
            func,
            args,
            kwargs,
            dtype=self.dtype,
            device_type=self.device_type,
        )


class TestDTensorOps(DistTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    # only allow float dytpe for now, we can relax this constraint
    # when feel necessary later (i.e when adding quantization support).
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @suppress_warnings
    @ops(common_ops.op_db, allowed_dtypes=(torch.float,))
    def test_dtensor_ops_dispatch(self, device, dtype, op):
        pg_backend = "nccl" if device == "cuda" else "gloo"
        if pg_backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.init_pg(backend=pg_backend)
        self.mesh = DeviceMesh(device, torch.arange(self.world_size))
        # test each op with dist tensor inputs and normal inputs
        func = op.get_op()
        if op.name in op_inputs_skips:
            # skip invalid input ops
            self.destroy_pg()
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            with DTensorCrossRefDispatchMode.push(
                self, dtype=dtype, device=device
            ):
                func(*args, **kwargs)
                # we need to figure out a way to test the out variant, out variant testing
                # is tricky, as we need to pre allocate the dtensor out, some of them rely
                # on sharding placements to be pre-known (i.e. mm.out)
                # if isinstance(expected, torch.Tensor) and op.supports_out:
                #     func(*args, **kwargs, out=expected)

        self.destroy_pg()


instantiate_device_type_tests(TestDTensorOps, globals())


def print_op_str_if_not_supported(op_str):
    op = OperatorName.parse(op_str)
    packet = getattr(torch.ops.aten, str(op.name))
    overload = getattr(
        packet, op.overload_name if op.overload_name else "default"
    )
    if any(
        overload in d
        for d in [dtensor_dispatch_skips, dtensor_dispatch_device_skips["cuda"]]
    ):
        print(f"{overload}  # SKIP")
    if any(
        overload in d
        for d in [
            dtensor_dispatch_expected_failures,
            dtensor_dispatch_device_expected_failures["cuda"],
        ]
    ):
        print(overload)


if __name__ == "__main__":
    COMPARE_TEXT = os.getenv("PYTORCH_COMPARE_TEXT", None)
    if COMPARE_TEXT is not None:
        with open(COMPARE_TEXT, "r") as f:
            for op_str in f:
                print_op_str_if_not_supported(op_str.strip())
        sys.exit(0)

    run_tests()
