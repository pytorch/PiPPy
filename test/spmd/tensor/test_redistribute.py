# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.distributed_c10d import ReduceOp

from torch.testing._internal.common_utils import run_tests

from ..test_utils import DistTensorTestBase, with_comms
from spmd.tensor import DeviceMesh, DTensor, Replicate, Shard, _Partial


class RedistributeTest(DistTensorTestBase):
    @with_comms
    def test_shard_to_replicate_forward_backward(self):
        # 1) test shard -> replicate forward
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_dim = 0
        shard_spec = [Shard(shard_dim)]
        replica_spec = [Replicate()]
        expected_tensor = torch.randn(
            12, 3, device=self.device_type, requires_grad=True
        )
        chunked_list = expected_tensor.chunk(self.world_size, shard_dim)
        # make local tensor as the element of the corresponding chunked list
        local_tensor = chunked_list[self.rank]
        # direct DTensor constructor should always be leaf
        sharded_tensor = DTensor(
            local_tensor, device_mesh, shard_spec, requires_grad=True
        )
        global_sharded_tensor = sharded_tensor.redistribute(
            device_mesh, replica_spec
        )
        self.assertEqual(global_sharded_tensor.size(), torch.Size([12, 3]))
        self.assertEqual(expected_tensor, global_sharded_tensor.to_local())

        # 2) test shard -> replicate backward:
        # should give gradient as shard
        local_grad = torch.ones(12, 3, device=self.device_type)
        grad_output = DTensor.from_local(local_grad, device_mesh, replica_spec)
        global_sharded_tensor.backward(grad_output)
        grad_input = sharded_tensor.grad
        self.assertEqual(grad_input.placements, shard_spec)
        self.assertEqual(grad_input.to_local(), torch.ones(3, 3))

    @with_comms
    def test_replicate_to_replicate_forward_backward(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]
        local_tensor = torch.randn(
            12, 3, device=self.device_type, requires_grad=True
        )
        # 1) test replicate -> replicate forward
        replica_tensor = DTensor(
            local_tensor, device_mesh, replica_spec, requires_grad=True
        )
        global_replica_tensor = replica_tensor.redistribute(
            device_mesh, replica_spec
        )
        self.assertEqual(replica_tensor.size(), local_tensor.size())
        self.assertEqual(replica_tensor, global_replica_tensor)

        # 2) test replicate -> replicate backward:
        # should give gradient as replicate
        local_grad = torch.ones(12, 3, device=self.device_type)
        grad_output = DTensor(local_grad, device_mesh, replica_spec)
        global_replica_tensor.backward(grad_output)
        grad_input = replica_tensor.grad
        self.assertEqual(grad_input.placements, replica_spec)
        self.assertEqual(grad_input.to_local(), local_grad)

    @with_comms
    def test_replicate_to_shard_forward_backward(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_dim = 0
        shard_spec = [Shard(shard_dim)]
        replica_spec = [Replicate()]
        # 1) test replicate -> shard forward
        local_replica = torch.randn(
            12, 3, device=self.device_type, requires_grad=True
        )
        chunked_list = local_replica.chunk(self.world_size, shard_dim)
        # make local tensor as the element of the corresponding chunked list
        local_tensor = chunked_list[self.rank]
        replica_tensor = DTensor(
            local_replica, device_mesh, replica_spec, requires_grad=True
        )
        reshard_tensor = replica_tensor.redistribute(device_mesh, shard_spec)
        self.assertEqual(reshard_tensor.size(), replica_tensor.size())
        self.assertEqual(reshard_tensor.placements, shard_spec)
        self.assertEqual(reshard_tensor.to_local(), local_tensor)

        # 2) test replicate -> shard backward:
        # should give gradient as replicate
        local_grad = torch.ones(3, 3, device=self.device_type)
        grad_output = DTensor.from_local(local_grad, device_mesh, shard_spec)
        reshard_tensor.backward(grad_output)
        grad_input = replica_tensor.grad
        self.assertEqual(grad_input.placements, replica_spec)
        self.assertEqual(grad_input.to_local(), torch.ones(12, 3))

    @with_comms
    def test_partial_to_replicate(self):
        # we only need to test forward path related to partial
        # becaues _Partial should only exist in op impls
        # and we don't allow reshard to produce a partial
        # placement (i.e. user can't reshard to partial)
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_local = torch.randn(
            12, 3, device=self.device_type, requires_grad=True
        )
        partial_spec = [_Partial(ReduceOp.SUM)]
        replica_spec = [Replicate()]
        # test partial -> replicate, which trigger all_reduce
        partial_tensor = DTensor(partial_local, device_mesh, partial_spec)
        global_partial_tensor = partial_tensor.redistribute(
            device_mesh, replica_spec
        )
        self.assertEqual(partial_tensor.size(), partial_local.size())
        self.assertEqual(partial_local * 4, global_partial_tensor.to_local())

    @with_comms
    def test_partial_to_shard_0(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_dim = 0
        shard_spec = [Shard(shard_dim)]
        partial_spec = [_Partial(ReduceOp.SUM)]
        partial_local = torch.ones(12, 3, device=self.device_type)
        partial_tensor = DTensor(partial_local, device_mesh, partial_spec)
        # test partial to shard 0, trigger reduce_scatter
        scatter_shard_tensor = partial_tensor.redistribute(
            device_mesh, shard_spec
        )
        self.assertEqual(scatter_shard_tensor.size(), partial_tensor.size())
        self.assertEqual(scatter_shard_tensor.placements, shard_spec)
        self.assertEqual(scatter_shard_tensor.to_local(), torch.ones(3, 3) * 4)

    @with_comms
    def test_partial_to_shard_1(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_dim = 1
        shard1_spec = [Shard(shard_dim)]
        partial_spec = [_Partial(ReduceOp.SUM)]
        partial_local = torch.ones(4, 12, device=self.device_type)
        # test partial to shard 1, trigger reduce_scatter
        partial_tensor = DTensor(partial_local, device_mesh, partial_spec)
        scatter_shard_tensor = partial_tensor.redistribute(
            device_mesh, shard1_spec
        )
        self.assertEqual(scatter_shard_tensor.size(), partial_tensor.size())
        self.assertEqual(scatter_shard_tensor.placements, shard1_spec)
        self.assertEqual(scatter_shard_tensor.to_local(), torch.ones(4, 3) * 4)


if __name__ == "__main__":
    run_tests()
