# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from spmd.test._utils import DistTensorTestBase, with_comms
from spmd import distribute_tensor, DeviceMesh, Shard, Replicate


class DistMatrixOpsTest(DistTensorTestBase):
    @with_comms
    def test_addmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_shard = torch.randn(12, 8)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        tensor_to_replicate = torch.randn(8, 4)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
        input_tensor = torch.randn(4)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        dist_res = torch.addmm(input, mat1, mat2)
        local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
        self.assertEqual(
            dist_res.redistribute(device_mesh, replica_spec).to_local(),
            local_res,
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
            dist_res.redistribute(device_mesh, replica_spec).to_local(),
            local_res,
        )

        # backward
        grad_res = torch.ones(12, 16)
        grad_dist_res = distribute_tensor(grad_res, device_mesh, shard_spec)
        dist_res.backward(grad_dist_res)
        # print(mat1.grad)
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
