# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.distributed_c10d import ReduceOp

from torch.testing._internal.common_utils import run_tests

from spmd.test._utils import DistTensorTestBase, with_comms  # type: ignore
from spmd.tensor import DeviceMesh, DTensor, Replicate, Shard, _Partial
from spmd import distribute_tensor
import functools


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


class MultiDimRedistributeTest(DistTensorTestBase):
    @property
    def world_size(self) -> int:
        return 6

    @with_comms
    def test_multi_dim_mesh(self):
        mesh_shape = torch.arange(self.world_size).view(-1, 2)
        device_mesh = DeviceMesh(self.device_type, mesh_shape)
        tensor_shape = (6, 12)


        if torch.distributed.get_rank() == 0:
            full_tensor = torch.randn(*tensor_shape)
        else:
            # these should be entirely ignored
            # because distribute_tensor is expected to override shards in ranks != 0
            full_tensor = torch.ones(*tensor_shape)


        possibilities = [Replicate()] + [Shard(i) for i in range(full_tensor.ndim)]

        import itertools
        all_outputs = list(itertools.product(*(mesh_shape.ndim * [possibilities])))
        all_inputs = list(itertools.product(*(mesh_shape.ndim * [possibilities + [_Partial()]])))

        for inputs in all_inputs:
            # if partial, temporarily make it Replicated, then replace replicated with partial afterwards
            repl_inputs = [Replicate() if s.is_partial() else s for s in inputs]
            dt = distribute_tensor(full_tensor, device_mesh, repl_inputs)

            if repl_inputs != inputs:
                # create a new DTensor reinterpreting some of the replicated entires as "Partial"
                dt = DTensor(dt.to_local(), device_mesh, inputs)

            if torch.distributed.get_rank() == 0:
                    print(inputs)

            for outputs in all_outputs:
                dt2 = dt.redistribute(device_mesh, outputs)
                if torch.distributed.get_rank() == 0:
                    print('  ', outputs)
                
                local_full = dt.redistribute(device_mesh, device_mesh.ndim * [Replicate()]).to_local()
                if torch.distributed.get_rank() == 0:
                    self.assertEqual(local_full.shape, full_tensor.shape)

                    num_sums = 1
                    for idx, input in enumerate(inputs):
                        if input.is_partial():
                            num_sums *= mesh_shape.size(idx)
                    expected = num_sums * full_tensor  # + torch.ones(*tensor_shape)
                    self.assertEqual(local_full, expected)


if __name__ == "__main__":
    run_tests()
