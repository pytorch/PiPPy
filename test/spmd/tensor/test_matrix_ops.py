# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from spmd.tensor.api import DTensor
from spmd.test._utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
    TEST_GPU_NUM,
)
from spmd import distribute_tensor, DeviceMesh
from spmd.tensor.placement_types import Shard, Replicate, _Partial


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
        local_res = torch.addmm(
            input_tensor, tensor_to_shard, tensor_to_replicate
        )
        self.assertEqual(
            dist_res.redistribute(device_mesh, replica_spec).to_local(),
            local_res,
        )

    @with_comms
    def test_addmm_auto_redistribute(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = [Shard(0)]
        shard1_spec = [Shard(1)]
        replica_spec = [Replicate()]

        tensor_to_shard1 = torch.randn(12, 8, requires_grad=True)
        mat1 = distribute_tensor(tensor_to_shard1, device_mesh, shard1_spec)
        tensor_to_shard0 = torch.randn(8, 4, requires_grad=True)
        mat2 = distribute_tensor(tensor_to_shard0, device_mesh, shard0_spec)
        input_tensor = torch.randn(4, requires_grad=True)
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        local_res = torch.addmm(
            input_tensor, tensor_to_shard1, tensor_to_shard0
        )
        dist_res = torch.addmm(input, mat1, mat2)

        # test if addmm output is a partial
        self.assertIsInstance(dist_res, DTensor)
        self.assertIsInstance(dist_res.placements[0], _Partial)

        # test if result is the same as tensor
        replica_res = dist_res.redistribute(device_mesh, replica_spec)
        dist_local_res = replica_res.to_local()
        self.assertEqual(local_res, dist_local_res)

        # backward checks
        dist_local_res.sum().backward()
        local_res.sum().backward()
        self.assertIsNotNone(mat2.grad)
        mat2_grad = mat2.grad.redistribute(device_mesh, replica_spec)
        self.assertEqual(mat2_grad.to_local(), tensor_to_shard0.grad)

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
        self.assertIsNotNone(mat1.grad)

    @with_comms
    def test_t(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_transpose = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_transpose, device_mesh, shard_spec)
        tranposed_mat = mat.t()
        self.assertEqual(tranposed_mat.size(), torch.Size([8, 12]))
        self.assertEqual(tranposed_mat.placements, [Shard(1)])
        tranposed_mat2 = tranposed_mat.t()
        self.assertEqual(tranposed_mat2.size(), torch.Size([12, 8]))
        self.assertEqual(tranposed_mat2.placements, shard_spec)

    # TODO: Need to investigate why test failed in CPU for baddbmm.
    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_baddbmm(self):
        # If beta is 0, input tensor will be ignored
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        batch_1 = torch.rand(3, 5, 8, device=self.device_type)
        batch_2 = torch.rand(3, 8, 6, device=self.device_type)
        local_result = torch.baddbmm(
            tensor, batch_1, batch_2, beta=0.0, alpha=0.5
        )
        sharding = [Shard(0)]
        tensor_dt = DTensor.from_local(tensor, device_mesh, sharding)
        batch_1_dt = DTensor.from_local(batch_1, device_mesh, sharding)
        batch_2_dt = DTensor.from_local(batch_2, device_mesh, sharding)
        new_dt = torch.baddbmm(
            tensor_dt, batch_1_dt, batch_2_dt, beta=0.0, alpha=0.5
        )
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        self.assertEqual(new_dt.to_local(), local_result)

        # If beta != 0
        local_result = torch.baddbmm(
            tensor, batch_1, batch_2, beta=0.8, alpha=0.5
        )
        new_dt = torch.baddbmm(
            tensor_dt, batch_1_dt, batch_2_dt, beta=0.8, alpha=0.5
        )
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        self.assertEqual(new_dt.to_local(), local_result)

    @with_comms
    def test_sharded_bmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        input = torch.rand(3, 5, 8, device=self.device_type)
        mat_2 = torch.rand(3, 8, 6, device=self.device_type)
        local_result = torch.bmm(input, mat_2)
        sharding = [Shard(0)]
        input_dt = DTensor.from_local(input, device_mesh, sharding)
        mat_2_dt = DTensor.from_local(mat_2, device_mesh, sharding)
        new_dt = torch.bmm(input_dt, mat_2_dt)
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        self.assertEqual(new_dt.to_local(), local_result)

if __name__ == "__main__":
    run_tests()
