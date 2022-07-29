# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.distributed.distributed_c10d import ReduceOp
from torch.testing._internal.common_utils import run_tests
from ..test_utils import DistTensorTestBase, with_comms
from spmd import (
    distribute_tensor,
    DeviceMesh,
    Shard,
    Replicate,
    _Partial,
    Tensor,
)


class DistMatrixOpsTest(DistTensorTestBase):
    @with_comms
    def test_addmm(self):
        from torch.distributed import get_rank

        rank = get_rank()
        world_size = self.world_size

        device_mesh = DeviceMesh(self.device_type, list(range(world_size)))
        placement_rep = [Replicate()]
        placement_par = [_Partial(ReduceOp.SUM)]
        placement_s0 = [Shard(0)]
        placement_s1 = [Shard(1)]

        # Ground truth
        torch.manual_seed(0)
        input = torch.randn((world_size, world_size))
        mat1 = torch.randn((world_size, world_size * 2))
        mat2 = torch.randn((world_size * 2, world_size))
        output = torch.addmm(input, mat1, mat2)

        # Replicate, Replicate, Replicate -> Replicate
        dist_input = Tensor.from_local(input, device_mesh, placement_rep)
        dist_mat1 = Tensor.from_local(mat1, device_mesh, placement_rep)
        dist_mat2 = Tensor.from_local(mat2, device_mesh, placement_rep)
        dist_output = torch.addmm(dist_input, dist_mat1, dist_mat2)
        self.assertEqual(dist_output.local_tensor(), output)

        # Shard(0), Shard(0), Replicate -> Shard(0)
        dist_input = Tensor.from_local(
            input[rank : rank + 1, :], device_mesh, placement_s0
        )
        dist_mat1 = Tensor.from_local(
            mat1[rank : rank + 1, :], device_mesh, placement_s0
        )
        dist_mat2 = Tensor.from_local(mat2, device_mesh, placement_rep)
        dist_output = torch.addmm(dist_input, dist_mat1, dist_mat2)
        self.assertEqual(dist_output.local_tensor(), output[rank : rank + 1])

        # Shard(1), Replicate, Shard(1) -> Shard(1)
        dist_input = Tensor.from_local(
            input[:, rank : rank + 1], device_mesh, placement_s1
        )
        dist_mat1 = Tensor.from_local(mat1, device_mesh, placement_rep)
        dist_mat2 = Tensor.from_local(
            mat2[:, rank : rank + 1], device_mesh, placement_s1
        )
        dist_output = torch.addmm(dist_input, dist_mat1, dist_mat2)
        self.assertEqual(dist_output.local_tensor(), output[:, rank : rank + 1])

        # Partial, Shard(1), Shard(0) -> Partial
        dist_input = Tensor.from_local(
            input if rank == 0 else torch.zeros_like(input),
            device_mesh,
            placement_par,
        )
        dist_mat1 = Tensor.from_local(
            mat1[:, rank * 2 : (rank + 1) * 2], device_mesh, placement_s1
        )
        dist_mat2 = Tensor.from_local(
            mat2[rank * 2 : (rank + 1) * 2, :], device_mesh, placement_s0
        )
        dist_output = torch.addmm(dist_input, dist_mat1, dist_mat2)
        self.assertEqual(
            dist_output.redistribute(device_mesh, placement_rep).local_tensor(),
            output,
        )

    @with_comms
    def test_mm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_shard = torch.randn(12, 8, requires_grad=True)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        tensor_to_replicate = torch.randn(8, 16, requires_grad=True)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)

        dist_res = torch.mm(mat1, mat2)
        local_res = torch.mm(tensor_to_shard, tensor_to_replicate)
        self.assertEqual(
            dist_res.redistribute(device_mesh, replica_spec).local_tensor(),
            local_res,
        )

        # backward
        grad_res = torch.ones(12, 16)
        grad_dist_res = distribute_tensor(grad_res, device_mesh, shard_spec)
        dist_res.backward(grad_dist_res)
        print(mat1.grad)
        # dist_res.sum().backward()

    @with_comms
    def test_t(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_transpose = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_transpose, device_mesh, shard_spec)
        tranposed_mat = mat.t()
        self.assertEqual(tranposed_mat.size(), torch.Size([8, 12]))
        self.assertEqual(tranposed_mat.placements, [Shard(1)])
        tranposed_mat2 = tranposed_mat.t()
        self.assertEqual(tranposed_mat2.size(), torch.Size([12, 8]))
        self.assertEqual(tranposed_mat2.placements, shard_spec)


if __name__ == "__main__":
    run_tests()
