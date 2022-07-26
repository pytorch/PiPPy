# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.distributed_c10d import ReduceOp

from torch.testing._internal.common_utils import run_tests
from ..test_utils import DistTensorTestBase, with_comms
from spmd.tensor import DeviceMesh, Tensor, Replicate, Shard, _Partial


class DistTensorTest(DistTensorTestBase):
    # @with_comms
    # def test_tensor_constructor(self):
    #     import spmd.tensor as dist_tensor
    #     shard_spec = PlacementSpec(device_mesh, strategies=[Shard(0)])
    #     empty_tensor = dist_tensor.empty((12, 10), placement_spec=shard_spec)
    #     zero_tensor = dist_tensor.zeros((12, 10), placement_spec=shard_spec)
    #     one_tensor = dist_tensor.ones((12, 10), placement_spec=shard_spec)

    #     zero_cuda_tensor = dist_tensor.zeros((12, 10), device="cuda", placement_spec=shard_spec)

    #     dist_tensor.empty_like(empty_tensor)
    #     dist_tensor.zero_like(empty_tensor)
    #     dist_tensor.one_like(empty_tensor)

    @with_comms
    def test_tensor_from_local(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = Tensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))

        replica_spec = [Replicate()]
        ddp_tensor = Tensor.from_local(local_tensor, device_mesh, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        partial_spec = [_Partial(ReduceOp.SUM)]
        partial_tensor = Tensor.from_local(local_tensor, device_mesh, partial_spec)
        self.assertEqual(partial_tensor.size(), local_tensor.size())

    @with_comms
    def test_placement_spec_read_only_after_set(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = Tensor.from_local(local_tensor, device_mesh, shard_spec)

        # modify shard_spec, and dist_tensor's spec should not be changed
        shard_spec[0] = Replicate()
        self.assertTrue(sharded_tensor.placements is not shard_spec)
        self.assertNotEqual(sharded_tensor.placements, shard_spec)

    @with_comms
    def test_tensor_properties(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = Tensor.from_local(local_tensor, device_mesh, shard_spec)
        print(sharded_tensor.device)


if __name__ == "__main__":
    run_tests()
