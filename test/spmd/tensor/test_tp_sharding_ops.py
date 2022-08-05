# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.placement_types import Replicate
import torch
from torch.testing._internal.common_utils import run_tests
from ..test_utils import DistTensorTestBase, with_comms, TEST_GPU_NUM
from spmd import DeviceMesh, Tensor, Shard, Replicate, distribute_tensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu


class TPShardingOpsTest(DistTensorTestBase):
    # TODO: We need to add CPU tests for ops in the future.
    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_view(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(0)
        tensor = torch.rand(16, 35, 26).cuda(self.rank)
        sharding = [Shard(0)]
        st = distribute_tensor(tensor, device_mesh, sharding).view(8, 4, 35, 13)
        st_new = distribute_tensor(tensor.view(8, 4, 35, 13), device_mesh, sharding)

        self.assertEqual(st.local_tensor(), st_new.local_tensor())
        self.assertEqual(st.placements[0], st_new.placements[0])

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_contiguous(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6).cuda(self.rank)
        sharding = [Shard(0)]
        dist_tensor = Tensor.from_local(tensor, device_mesh, sharding)
        self.assertTrue(dist_tensor.is_contiguous())
        new_dt = dist_tensor.transpose(0, 2)
        # TODO: Investigate why after transpose dt is still contiguous.
        self.assertFalse(new_dt.local_tensor().is_contiguous())
        new_dt = new_dt.contiguous()
        # TODO: Investigate why after contiguous dt is not contiguous.
        # self.assertTrue(new_dt.is_contiguous())
        self.assertTrue(new_dt.local_tensor().is_contiguous())

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_transpose(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6).cuda(self.rank)
        sharding = [Shard(0)]
        dist_tensor = Tensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.transpose(0, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=2))
        self.assertEqual(new_dt.local_tensor(), tensor.transpose(0, 2))
        new_dt = dist_tensor.transpose(1, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        self.assertEqual(new_dt.local_tensor(), tensor.transpose(1, 2))

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_baddbmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6).cuda(self.rank)
        batch_1 = torch.rand(3, 5, 8).cuda(self.rank)
        batch_2 = torch.rand(3, 8, 6).cuda(self.rank)
        local_result = torch.baddbmm(tensor, batch_1, batch_2, beta=0.0, alpha=0.5)
        sharding = [Shard(0)]
        tensor_dt = Tensor.from_local(tensor, device_mesh, sharding)
        batch_1_dt = Tensor.from_local(batch_1, device_mesh, sharding)
        batch_2_dt = Tensor.from_local(batch_2, device_mesh, sharding)
        new_dt = torch.baddbmm(tensor_dt, batch_1_dt, batch_2_dt, beta=0.0, alpha=0.5)
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        self.assertEqual(new_dt.local_tensor(), local_result)

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_bmm(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        input = torch.rand(3, 5, 8).cuda(self.rank)
        mat_2 = torch.rand(3, 8, 6).cuda(self.rank)
        local_result = torch.bmm(input, mat_2)
        sharding = [Shard(0)]
        input_dt = Tensor.from_local(input, device_mesh, sharding)
        mat_2_dt = Tensor.from_local(mat_2, device_mesh, sharding)
        new_dt = torch.bmm(input_dt, mat_2_dt)
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        self.assertEqual(new_dt.local_tensor(), local_result)

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_softmax(self):
        for softmax_dim in [1, 2, -1]:
            device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
            torch.manual_seed(self.rank)
            input = torch.rand(15, 27, 16).cuda(self.rank)
            local_result = torch.nn.functional.softmax(
                input, dim=softmax_dim, dtype=torch.float32
            )
            sharding = [Shard(0)]
            input_dt = Tensor.from_local(input, device_mesh, sharding)
            new_dt = torch.nn.functional.softmax(
                input_dt, dim=softmax_dim, dtype=torch.float32
            )
            self.assertTrue(new_dt.placements[0].is_shard(dim=0))
            self.assertEqual(new_dt.local_tensor(), local_result)

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_permute(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6).cuda(self.rank)
        sharding = [Shard(0)]
        dist_tensor = Tensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.permute(1, 0, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=1))
        self.assertEqual(new_dt.local_tensor(), tensor.permute(1, 0, 2))

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_replicated_permute(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(0)
        tensor = torch.rand(3, 5, 6).cuda(self.rank)
        sharding = [Replicate()]
        dist_tensor = Tensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.permute(1, 0, 2)
        self.assertTrue(new_dt.placements[0].is_replicate())
        self.assertEqual(new_dt.local_tensor(), tensor.permute(1, 0, 2))
        if self.rank == 0:
            print(new_dt.stride())
            print(tensor.permute(1, 0, 2).stride())
        self.assertEqual(new_dt.stride(), tensor.permute(1, 0, 2).stride())

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_cat(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor_1 = torch.rand(3, 5, 6).cuda(self.rank)
        tensor_2 = torch.rand(3, 5, 6).cuda(self.rank)
        tensor_3 = torch.rand(3, 5, 6).cuda(self.rank)
        sharding = [Shard(0)]
        dt_1 = Tensor.from_local(tensor_1, device_mesh, sharding)
        dt_2 = Tensor.from_local(tensor_2, device_mesh, sharding)
        dt_3 = Tensor.from_local(tensor_3, device_mesh, sharding)
        new_dt = torch.cat([dt_1, dt_2, dt_3])
        cat_dt = Tensor.from_local(
            torch.cat([tensor_1, tensor_2, tensor_3]), device_mesh, sharding
        )
        self.assertEqual(new_dt.local_tensor(), cat_dt.local_tensor())
        self.assertEqual(new_dt.size(), cat_dt.size())

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_sharded_split(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6).cuda(self.rank)
        sharding = [Shard(2)]
        dist_tensor = Tensor.from_local(tensor, device_mesh, sharding)
        dt_list = dist_tensor.split(dist_tensor.size(-1) // 2, dim=-1)
        local_tensors = tensor.split(3, dim=-1)
        for idx, dt in enumerate(dt_list):
            self.assertTrue(dt.placements[0].is_shard(dim=2))
            self.assertEqual(dt.local_tensor(), local_tensors[idx])

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_view_with_sharding_dim_change(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6).cuda(self.rank)
        sharding = [Shard(2)]
        dt = Tensor.from_local(tensor, device_mesh, sharding)
        dt = dt._view_with_sharding_dim_change(1, (3, -1, 6))
        self.assertTrue(dt.placements[0].is_shard(dim=1))
        self.assertEqual(dt.local_tensor(), tensor.view(3, -1, 6))


if __name__ == "__main__":
    run_tests()
