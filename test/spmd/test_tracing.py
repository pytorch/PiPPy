# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.distributed_c10d import (
    get_global_rank,
    get_world_size,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd.tensor import DeviceMesh

from typing import List


class TraceDeviceMeshTest(DistTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_tracing_all_reduce_nd(self):
        mesh_tensor = torch.arange(4).reshape(2, 2)
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
                reduced_tensor = mesh.all_reduce(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return reduced_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            reduced_tensor = traced_fn(local_tensor)
            res_num = sum(global_ranks)
            self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_broadcast_nd(self):
        mesh_tensor = torch.arange(4).reshape(2, 2)
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
                received_tensor = mesh.broadcast(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            received_tensor = traced_fn(local_tensor)
            res_num = global_ranks[0]
            self.assertEqual(received_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_scatter_nd(self):
        mesh_tensor = torch.arange(4).reshape(2, 2)
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

            def fn(tensors: List[torch.Tensor]):
                received_tensor = mesh.scatter(tensors, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)([t + 1 for t in scattered_tensors])

            received_tensor = traced_fn(scattered_tensors)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)

    @with_comms
    def test_all_gather_nd(self):
        mesh_tensor = torch.arange(4).reshape(2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # each rank have its own tensor, all_gather gives a list
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                gathered_tensors = mesh.all_gather(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return [t * 1 for t in gathered_tensors]

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            gathered_tensors = traced_fn(local_tensor)
            self.assertEqual(len(gathered_tensors), dim_group_size)
            for idx, gathered_tensor in enumerate(gathered_tensors):
                self.assertEqual(
                    gathered_tensor, torch.ones(3, 3) * global_ranks[idx]
                )


if __name__ == "__main__":
    run_tests()
