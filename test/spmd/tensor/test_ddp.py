# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from spmd.test._utils import DistTensorTestBase, with_comms  # type: ignore
from spmd import (
    distribute_tensor,
    distribute_module,
    DeviceMesh,
    DTensor,
    Shard,
    Replicate,
)


class MyModel(nn.Module):
    def __init__(self, n_features, n_layers, device):
        super().__init__()
        self.seq = nn.Sequential(
            *[
                nn.Linear(n_features, n_features, device=device)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        return self.seq(x)

    def reset_parameters(self):
        for m in self.seq:
            m.reset_parameters()


class DistTensorAPITest(DistTensorTestBase):
    @with_comms
    def test_distribute_tensor(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        tensor_to_shard = torch.randn(12, 3)
        sharded_tensor = distribute_tensor(
            tensor_to_shard, device_mesh, shard_spec
        )
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))
        local_tensor = sharded_tensor.to_local()
        self.assertEqual(local_tensor.size(), torch.Size([3, 3]))

    @with_comms
    def test_distribute_tensor_requires_grad(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        tensor_to_shard = torch.randn(12, 3, requires_grad=True)
        sharded_tensor = distribute_tensor(
            tensor_to_shard, device_mesh, shard_spec
        )
        self.assertTrue(sharded_tensor.requires_grad)
        self.assertTrue(sharded_tensor.is_leaf)
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))

    @with_comms
    def test_distribute_module(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        module_to_shard = MyModel(20, 20, device=self.device_type)
        shard_spec = [Shard(0)]
        sharded_module = distribute_module(
            module_to_shard, device_mesh, shard_spec
        )

        module_to_replicate = MyModel(20, 20, device=self.device_type)
        replica_spec = [Replicate()]
        replica_module = distribute_module(
            module_to_replicate, device_mesh, replica_spec
        )


class DDPWithDistTensorAPITest(DistTensorTestBase):
    @with_comms
    def test_full_replicated(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        n_features = 10
        model = MyModel(n_features, n_features, device=self.device_type)
        # mark model as replication
        replica_spec = [Replicate()]
        replicated_model = distribute_module(model, device_mesh, replica_spec)

        input = torch.randn(10, n_features, requires_grad=True)
        # mark input as replicated
        replicated_input = DTensor.from_local(input, device_mesh, replica_spec)

        output = model(replicated_input)
        output.sum().backward()
        param_grad = list(model.parameters())[0].grad
        self.assertTrue(isinstance(param_grad, DTensor))
        self.assertTrue(isinstance(param_grad.placements[0], Replicate))

    @with_comms
    def test_ddp_dist_tensor(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        n_features = 100
        model = MyModel(n_features, 1, device=self.device_type)
        # mark model as replication
        replica_spec = [Replicate()]
        replicated_model = distribute_module(model, device_mesh, replica_spec)

        shard0_spec = [Shard(0)]
        input = torch.randn(10, n_features)
        # mark input as shard on dim 0
        sharded_input = DTensor.from_local(input, device_mesh, shard0_spec)

        # run DDP like a normal model
        output = replicated_model(sharded_input)


if __name__ == "__main__":
    run_tests()
