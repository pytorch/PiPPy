# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
from torch.distributed.distributed_c10d import get_global_rank, get_world_size
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import run_tests

from spmd.api import SPMD, Schema
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd.tensor import (
    DeviceMesh,
    Replicate,
)

from copy import deepcopy


class TraceDeviceMeshTestBase:
    def _test_tracing_all_reduce_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                in_tensor = tensor.clone()
                mesh.all_reduce(in_tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.

                return in_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)
            print(f">>> traced graph: {traced_fn}")

            # execute traced DeviceMesh communication
            reduced_tensor = traced_fn(local_tensor.clone())
            # res_num = sum(global_ranks)
            # self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)

    def _test_broadcast_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                received_tensor = mesh.broadcast(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            received_tensor = traced_fn(local_tensor)
            res_num = global_ranks[0]
            self.assertEqual(received_tensor, torch.ones(3, 3) * res_num)

    def _test_scatter_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]
            tensor_to_scatter = torch.cat(scattered_tensors)

            def fn(tensor_to_scatter: torch.Tensor):
                received_tensor = mesh.scatter(tensor_to_scatter, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(tensor_to_scatter + 1)

            received_tensor = traced_fn(tensor_to_scatter)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)

    def _test_all_gather_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # each rank have its own tensor, all_gather gives a big tensor
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor, output_shape):
                return mesh.all_gather(tensor, output_shape, mesh_dim=dim)

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1, (dim_group_size * 3, 3))
            gathered_tensor = traced_fn(local_tensor, (dim_group_size * 3, 3))

            exp_tensor = torch.ones(3 * dim_group_size, 3)
            for i in range(len(global_ranks)):
                exp_tensor[i * 3 : (i + 1) * 3] = (
                    torch.ones(3, 3) * global_ranks[i]
                )
            self.assertEqual(gathered_tensor, exp_tensor)


class TraceDeviceMesh3DTest(DistTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_broadcast_nd(self):
        self._test_broadcast_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_scatter_nd(self):
        self._test_scatter_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_all_gather_nd(self):
        self._test_all_gather_nd(torch.arange(8).reshape(2, 2, 2))


class TraceDeviceMesh2DTest(DistTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_broadcast_nd(self):
        self._test_broadcast_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_scatter_nd(self):
        self._test_scatter_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_all_gather_nd(self):
        self._test_all_gather_nd(torch.arange(4).reshape(2, 2))


class TraceModuleTest(DistTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _test_trace_replicate(self, model, x):
        # if x.device.type == "cuda":
        ddp = DDP(deepcopy(model))
        spmd = SPMD(
            deepcopy(model),
            schema=Schema(
                mesh=DeviceMesh(
                    self.device_type, torch.arange(self.world_size)
                ),
                placements=[Replicate()],
            ),
        )

        ddp(x).sum().backward()
        spmd(x).sum().backward()
        for p1, p2 in zip(ddp.parameters(), spmd.parameters()):
            # DDP divides gradients by world size to compute average, but
            # _Partial tensor shouldn't do that automatically. Hence explicitly
            # do division here.
            self.assertTrue(p1.grad.allclose(p2.grad / self.world_size))

    @with_comms
    def test_sequential(self):
        model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)]).to(
            self.device_type
        )
        x = torch.randn(2, 10).to(self.device_type)
        self._test_trace_replicate(model, x)

    @with_comms
    def test_parallel(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.module_list = nn.ModuleList(
                    [nn.Linear(10, 10) for _ in range(2)]
                )

            def forward(self, x):
                return sum([m(x) for m in self.module_list])

        model = Model().to(self.device_type)
        x = torch.randn(2, 10).to(self.device_type)
        self._test_trace_replicate(model, x)


if __name__ == "__main__":
    run_tests()
