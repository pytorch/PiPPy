# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd import DeviceMesh, DTensor, Shard, Replicate, distribute_tensor


class TPShardingOpsTest(DistTensorTestBase):
    @with_comms
    def test_sharded_view(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(0)
        tensor = torch.rand(16, 35, 26)
        sharding = [Shard(0)]
        st = distribute_tensor(tensor, device_mesh, sharding).view(8, 4, 35, 13)
        st_new = distribute_tensor(
            tensor.view(8, 4, 35, 13), device_mesh, sharding
        )
        self.assertEqual(st.to_local(), st_new.to_local())
        self.assertEqual(st.placements[0], st_new.placements[0])
        self.assertEqual(st.stride(), tensor.view(8, 4, 35, 13).stride())
        self.assertEqual(st_new.stride(), tensor.view(8, 4, 35, 13).stride())

    @with_comms
    def test_sharded_transpose(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.transpose(0, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=2))
        self.assertEqual(new_dt.to_local(), tensor.transpose(0, 2))
        new_dt = dist_tensor.transpose(1, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        self.assertEqual(new_dt.to_local(), tensor.transpose(1, 2))

    @with_comms
    def test_sharded_softmax_with_bwd(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(10)
        tensor = torch.rand(
            16, 5, 8, device=self.device_type, requires_grad=True
        )
        local_result = tensor.softmax(dim=-1)
        sharding = [Shard(0)]
        tensor_dt = distribute_tensor(tensor, device_mesh, sharding)
        new_dt = tensor_dt.softmax(dim=-1)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(tensor_dt.requires_grad)
        self.assertTrue(new_dt.placements[0].is_shard(dim=0))
        local_result.sum().backward()
        new_dt.sum().backward()
        tensor_dt_grad = torch.empty(
            tensor_dt.grad.size(),
            dtype=tensor_dt.grad.dtype,
            device=self.device_type,
        )
        dist._all_gather_base(tensor_dt_grad, tensor_dt.grad.to_local())
        self.assertEqual(tensor_dt_grad, tensor.grad)

    @with_comms
    def test_sharded_permute(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.permute(1, 0, 2)
        self.assertTrue(new_dt.placements[0].is_shard(dim=1))
        self.assertEqual(new_dt.to_local(), tensor.permute(1, 0, 2))

    @with_comms
    def test_replicated_permute(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(0)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Replicate()]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        new_dt = dist_tensor.permute(1, 0, 2)
        self.assertTrue(new_dt.placements[0].is_replicate())
        self.assertEqual(new_dt.to_local(), tensor.permute(1, 0, 2))
        self.assertEqual(new_dt.stride(), tensor.permute(1, 0, 2).stride())

    @with_comms
    def test_sharded_cat(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor_1 = torch.rand(3, 5, 6)
        tensor_2 = torch.rand(3, 5, 6)
        tensor_3 = torch.rand(3, 5, 6)
        sharding = [Shard(0)]
        dt_1 = DTensor.from_local(tensor_1, device_mesh, sharding)
        dt_2 = DTensor.from_local(tensor_2, device_mesh, sharding)
        dt_3 = DTensor.from_local(tensor_3, device_mesh, sharding)
        new_dt = torch.cat([dt_1, dt_2, dt_3])
        cat_dt = DTensor.from_local(
            torch.cat([tensor_1, tensor_2, tensor_3]), device_mesh, sharding
        )
        self.assertEqual(new_dt.to_local(), cat_dt.to_local())
        self.assertEqual(new_dt.size(), cat_dt.size())

    @with_comms
    def test_sharded_split(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Shard(2)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        dt_list = dist_tensor.split(dist_tensor.size(-1) // 2, dim=-1)
        local_tensors = tensor.split(3, dim=-1)
        for idx, dt in enumerate(dt_list):
            self.assertTrue(dt.placements[0].is_shard(dim=2))
            self.assertEqual(dt.to_local(), local_tensors[idx])

    @with_comms
    def test_view_with_sharding_dim_change(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        tensor = torch.rand(3, 5, 6, device=self.device_type)
        sharding = [Shard(2)]
        dt = DTensor.from_local(tensor, device_mesh, sharding)
        dt = dt._view_with_sharding_dim_change(1, (3, -1, 6))
        self.assertTrue(dt.placements[0].is_shard(dim=1))
        self.assertEqual(dt.to_local(), tensor.view(3, -1, 6))


if __name__ == "__main__":
    run_tests()
