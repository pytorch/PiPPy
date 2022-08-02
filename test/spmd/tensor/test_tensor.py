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
        sharded_tensor = Tensor.from_local(
            local_tensor, device_mesh, shard_spec
        )
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))

        replica_spec = [Replicate()]
        ddp_tensor = Tensor.from_local(local_tensor, device_mesh, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        partial_spec = [_Partial(ReduceOp.SUM)]
        partial_tensor = Tensor.from_local(
            local_tensor, device_mesh, partial_spec
        )
        self.assertEqual(partial_tensor.size(), local_tensor.size())

        # test dist tensor works with torch.Tensor during backwards
        local_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        # do some operations on local tensor
        local_tensor_temp = local_tensor_with_grad * 3
        # create the dist tensor with non leaf local tensor, dist tensor created
        # should also be non leaf node
        dist_tensor = Tensor.from_local(
            local_tensor_temp, device_mesh, shard_spec
        )
        self.assertFalse(dist_tensor.is_leaf)
        # do some random operations on dist tensor
        output = dist_tensor * 3
        self.assertIsInstance(output, Tensor)
        # trigger .backward() on dist tensor directly
        local_grad = torch.ones(3, 3)
        grad_output = Tensor.from_local(local_grad, device_mesh, shard_spec)
        # run backward directly on dist tensor
        output.backward(grad_output)
        # check it gradients flow back to original torch.Tensor
        self.assertIsNotNone(local_tensor_with_grad.grad)
        expected_grad = torch.ones(3, 3) * 9
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    def test_local_tensor(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor_with_grad = torch.randn(3, 3, requires_grad=True)

        sharded_tensor = Tensor.from_local(
            local_tensor_with_grad, device_mesh, shard_spec
        )
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))
        self.assertEqual(sharded_tensor.local_tensor(), local_tensor_with_grad)

        # test dist tensor works with torch.Tensor during backwards
        # dist tensor created is a leaf node, do some operation on dist tensor
        temp_st = sharded_tensor * 3

        # do some operation on local tensor of the dist tensor
        new_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        res = temp_st.local_tensor() + new_tensor_with_grad
        # call backward directly on torch.Tensor, and see if it works by
        # propagating through dist tensor
        res.sum().backward()
        self.assertIsNotNone(sharded_tensor.grad)

        expected_grad = Tensor.from_local(
            torch.ones(3, 3) * 3, device_mesh, shard_spec
        )
        self.assertEqual(
            sharded_tensor.grad.local_tensor(), expected_grad.local_tensor()
        )

    @with_comms
    def test_from_local_then_local_tensor(self):
        # this test ensure end to end from torch.Tensor -> dist tensor -> torch.Tensor works
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        # step 1. construct from construct local tensor
        local_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        # do some operations on local tensor
        local_tensor_temp = local_tensor_with_grad + 8
        # step 2. create the dist tensor with non leaf local tensor, dist tensor
        # created should also be non leaf node
        dist_tensor = Tensor.from_local(
            local_tensor_temp, device_mesh, shard_spec
        )
        self.assertFalse(dist_tensor.is_leaf)
        # do some random operations on dist tensor
        output = dist_tensor * 6
        self.assertIsInstance(output, Tensor)

        # step 3. do some operation on local tensor of the dist tensor
        new_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        res = output.local_tensor() + new_tensor_with_grad
        # call backward directly on torch.Tensor, and see if it works by
        # propagating all the way back to the original torch.Tensor
        res.sum().backward()
        self.assertIsNotNone(local_tensor_with_grad.grad)

        expected_grad = torch.ones(3, 3) * 6
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    def test_placement_spec_read_only_after_set(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = Tensor.from_local(
            local_tensor, device_mesh, shard_spec
        )

        # modify shard_spec, and dist_tensor's spec should not be changed
        shard_spec[0] = Replicate()
        self.assertTrue(sharded_tensor.placements is not shard_spec)
        self.assertNotEqual(sharded_tensor.placements, shard_spec)

    @with_comms
    def test_tensor_properties(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = Tensor.from_local(
            local_tensor, device_mesh, shard_spec
        )
        self.assertEqual(str(sharded_tensor.device), self.device_type)


if __name__ == "__main__":
    run_tests()
