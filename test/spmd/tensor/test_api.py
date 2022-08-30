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


class DTensorAPITest(DistTensorTestBase):
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
        # fully shard all linear modules on dim 0
        module_to_shard = MyModel(20, 20, device=self.device_type)
        shard_spec = [Shard(0)]

        def shard_fn(name, module):
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, shard_spec)
                    )
                    module.register_parameter(name, dist_param)

        sharded_module = distribute_module(
            module_to_shard, device_mesh, shard_fn
        )
        for param in sharded_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, shard_spec)

        replica_spec = [Replicate()]
        # fully replicate all modules without passing in partition_fn
        module_to_replicate = MyModel(20, 20, device=self.device_type)
        replica_module = distribute_module(module_to_replicate, device_mesh)
        for param in replica_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, replica_spec)

        # fully replicate all modules by passing in partition_fn
        def replicate_fn(name, module):
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, replica_spec)
                    )
                    module.register_parameter(name, dist_param)

        module_to_replicate = MyModel(20, 20, device=self.device_type)
        replica_module = distribute_module(
            module_to_replicate, device_mesh, replicate_fn
        )
        for param in replica_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, replica_spec)

        # only shard part of module, and rest of module should be replicate
        def shard_fn(name, module):
            if isinstance(module, nn.Linear) and (
                name == "seq.0" or name == "seq.8"
            ):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, shard_spec)
                    )
                    module.register_parameter(name, dist_param)

        module_to_distribute = MyModel(20, 20, device=self.device_type)
        dist_module = distribute_module(
            module_to_distribute, device_mesh, shard_fn
        )
        for name, param in dist_module.named_parameters():
            self.assertIsInstance(param, DTensor)
            if name.startswith("seq.0") or name.startswith("seq.8"):
                self.assertEqual(param.placements, shard_spec)
            else:
                self.assertEqual(param.placements, replica_spec)

    @with_comms
    def test_distribute_module_input_fn_output_fn(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # fully replicate all linear modules
        module_to_replicate = MyModel(20, 1, device=self.device_type)

        # mark input sharding on dim 0
        def input_fn(inputs):
            return DTensor.from_local(inputs[0], device_mesh, [Shard(0)])

        def output_fn(outputs):
            assert isinstance(outputs, DTensor)
            return outputs.to_local()

        replica_module = distribute_module(
            module_to_replicate,
            device_mesh,
            input_fn=input_fn,
            output_fn=output_fn,
        )

        input_tensor = torch.randn(5, 20, device=self.device_type)
        local_out = replica_module(input_tensor)
        self.assertIsInstance(local_out, torch.Tensor)
        self.assertNotIsInstance(local_out, DTensor)

        # full replicate (even on inputs)
        model = MyModel(10, 10, device=self.device_type)

        def replicate_input_fn(inputs):
            return DTensor.from_local(inputs[0], device_mesh, [Replicate()])

        replica_model = distribute_module(
            model,
            device_mesh,
            input_fn=replicate_input_fn,
        )
        input = torch.randn(10, 10, requires_grad=True)
        output = replica_model(input)
        output.sum().backward()
        param_grad = list(replica_model.parameters())[0].grad
        self.assertTrue(isinstance(param_grad, DTensor))
        self.assertTrue(isinstance(param_grad.placements[0], Replicate))


if __name__ == "__main__":
    run_tests()
