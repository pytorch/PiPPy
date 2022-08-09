# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.distributed_c10d import ReduceOp

from torch.testing._internal.common_utils import run_tests
from ..test_utils import DistTensorTestBase, with_comms
from spmd.tensor import DeviceMesh, DTensor, Replicate, Shard, _Partial


class DistTensorTest(DistTensorTestBase):

    @with_comms
    def test_tensor_constructor(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3, requires_grad=True)
        dist_tensor = DTensor(
            local_tensor, device_mesh, shard_spec, requires_grad=True
        )
        self.assertEqual(dist_tensor.size(), torch.Size((12, 3)))

        with self.assertWarnsRegex(UserWarning, "To construct"):
            DTensor(local_tensor, device_mesh, shard_spec)

        local_tensor = torch.randn(3, 3, requires_grad=False)
        with self.assertWarnsRegex(UserWarning, "To construct"):
            dist_tensor = DTensor(
                local_tensor, device_mesh, shard_spec, requires_grad=True
            )

    @with_comms
    def test_tensor_stride(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = [Shard(0)]
        local_tensor = torch.randn(4, 8)
        dist_tensor = DTensor(local_tensor, device_mesh, shard0_spec)
        # won't affect stride
        self.assertEqual(dist_tensor.stride(), (8, 1))

        shard1_spec = [Shard(1)]
        local_tensor = torch.randn(8, 4)
        dist_tensor = DTensor(local_tensor, device_mesh, shard1_spec)
        # will affect stride after DT initialized
        self.assertEqual(dist_tensor.stride(), (16, 1))

        # if initialized from a transposed mat
        local_tensor = torch.randn(8, 4, 8)
        local_tensor_t = local_tensor.permute(1, 2, 0)
        self.assertEqual(local_tensor_t.stride(), (8, 1, 32))
        dist_tensor = DTensor(local_tensor_t, device_mesh, shard1_spec)
        self.assertEqual(dist_tensor.stride(), (32, 1, 128))

    @with_comms
    def test_tensor_from_local(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(
            local_tensor, device_mesh, shard_spec
        )
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))

        replica_spec = [Replicate()]
        ddp_tensor = DTensor.from_local(local_tensor, device_mesh, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        partial_spec = [_Partial(ReduceOp.SUM)]
        partial_tensor = DTensor.from_local(
            local_tensor, device_mesh, partial_spec
        )
        self.assertEqual(partial_tensor.size(), local_tensor.size())

        # test dist tensor works with torch.Tensor during backwards
        local_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        # do some operations on local tensor
        local_tensor_temp = local_tensor_with_grad * 3
        # create the dist tensor with non leaf local tensor, dist tensor created
        # should also be non leaf node
        dist_tensor = DTensor.from_local(
            local_tensor_temp, device_mesh, shard_spec
        )
        self.assertFalse(dist_tensor.is_leaf)
        # do some random operations on dist tensor
        output = dist_tensor * 3
        self.assertIsInstance(output, DTensor)
        # trigger .backward() on dist tensor directly
        local_grad = torch.ones(3, 3)
        grad_output = DTensor.from_local(local_grad, device_mesh, shard_spec)
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
        local_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )

        sharded_tensor = DTensor(
            local_tensor_with_grad, device_mesh, shard_spec, requires_grad=True
        )
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))
        self.assertEqual(sharded_tensor.to_local(), local_tensor_with_grad)

        # test dist tensor works with torch.Tensor during backwards
        # dist tensor created is a leaf node, do some operation on dist tensor
        temp_st = sharded_tensor * 3

        # do some operation on local tensor of the dist tensor
        new_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        res = temp_st.to_local() + new_tensor_with_grad
        # call backward directly on torch.Tensor, and see if it works by
        # propagating through dist tensor
        res.sum().backward()
        self.assertIsNotNone(sharded_tensor.grad)

        expected_grad = DTensor(torch.ones(3, 3) * 3, device_mesh, shard_spec)
        self.assertEqual(
            sharded_tensor.grad.to_local(), expected_grad.to_local()
        )

    @with_comms
    def test_from_local_then_local_tensor(self):
        # this test ensure end to end from torch.Tensor -> dist tensor -> torch.Tensor works
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        # step 1. construct from construct local tensor
        local_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        # do some operations on local tensor
        local_tensor_temp = local_tensor_with_grad + 8
        # step 2. create the dist tensor with non leaf local tensor, dist tensor
        # created should also be non leaf node
        dist_tensor = DTensor.from_local(
            local_tensor_temp, device_mesh, shard_spec
        )
        self.assertFalse(dist_tensor.is_leaf)
        # do some random operations on dist tensor
        output = dist_tensor * 6
        self.assertIsInstance(output, DTensor)

        # step 3. do some operation on local tensor of the dist tensor
        new_tensor_with_grad = torch.randn(
            3, 3, device=self.device_type, requires_grad=True
        )
        res = output.to_local() + new_tensor_with_grad
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
        sharded_tensor = DTensor.from_local(
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
        sharded_tensor = DTensor.from_local(
            local_tensor, device_mesh, shard_spec
        )
        self.assertEqual(sharded_tensor.device.type, self.device_type)


if __name__ == "__main__":
    run_tests()
