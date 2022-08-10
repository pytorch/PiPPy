# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from ..test_utils import DistTensorTestBase, with_comms
from spmd import distribute_tensor, DeviceMesh, DTensor, Shard, Replicate


class DistTensorOpsTest(DistTensorTestBase):
    @with_comms
    def test_detach(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        detached_mat = mat.detach()
        self.assertFalse(detached_mat is mat)

    @with_comms
    def test_clone(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        specs = [[Replicate()], [Shard(0)]]
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        for spec in specs:
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            cloned_mat = mat.clone()
            self.assertFalse(cloned_mat is mat)
            self.assertEqual(cloned_mat.to_local(),  mat.to_local())

    @with_comms
    def test_contiguous(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(3, 5, 6)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        self.assertTrue(dist_tensor.is_contiguous())

        new_dt = dist_tensor.transpose(0, 2)
        self.assertFalse(new_dt.is_contiguous())
        self.assertFalse(new_dt.to_local().is_contiguous())

        new_dt = new_dt.contiguous()
        self.assertTrue(new_dt.is_contiguous())
        self.assertTrue(new_dt.to_local().is_contiguous())

    @with_comms
    def test_ones_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(4, 8)
        self.assertEqual(ones_expected, ones_like_dt.to_local())


if __name__ == "__main__":
    run_tests()
