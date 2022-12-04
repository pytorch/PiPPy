from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    DEVICE_TYPE,
)
import torch

from torch.distributed._tensor import DeviceMesh, distribute_tensor
from torch.distributed._tensor.placement_types import (
    Replicate,
    Shard,
    _Partial,
)
from spmd.tensor.experimental_ops import *  # noqa: F401 F403


class TraceModuleTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def test_native_layernorm(self):
        self.init_pg(backend="nccl" if DEVICE_TYPE == "cuda" else "gloo")
        mesh = DeviceMesh(DEVICE_TYPE, torch.arange(self.world_size))

        def func(x, w, b):
            y, _, _ = torch.native_layer_norm(x, x.shape[1:], w, b, eps=0.001)
            z = torch.mul(y, x)
            l = z.sum()
            gx, gw, gb = torch.autograd.grad(l, [x, w, b])
            return l, gx, gw, gb

        x = torch.randn([8, 16], device=DEVICE_TYPE, requires_grad=True)
        w = torch.randn([16], device=DEVICE_TYPE, requires_grad=True)
        b = torch.randn([16], device=DEVICE_TYPE, requires_grad=True)

        l, gx, gw, gb = func(x, w, b)

        d_x = distribute_tensor(x, mesh, [Shard(0)])
        d_w = distribute_tensor(w, mesh, [Replicate()])
        d_b = distribute_tensor(b, mesh, [Replicate()])

        d_l, d_gx, d_gw, d_gb = func(d_x, d_w, d_b)

        self.assertEqual(d_l.placements, [_Partial()])
        self.assertEqual(d_gx.placements, [Shard(0)])
        self.assertEqual(d_gw.placements, [_Partial()])
        self.assertEqual(d_gb.placements, [_Partial()])

        self.assertTrue(
            l.allclose(d_l.redistribute(mesh, [Replicate()]).to_local())
        )
        self.assertTrue(
            gx.allclose(d_gx.redistribute(mesh, [Replicate()]).to_local())
        )
        self.assertTrue(
            gw.allclose(d_gw.redistribute(mesh, [Replicate()]).to_local())
        )
        self.assertTrue(
            gb.allclose(d_gb.redistribute(mesh, [Replicate()]).to_local())
        )
