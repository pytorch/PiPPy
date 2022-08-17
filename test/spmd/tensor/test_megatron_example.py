# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
import torch.distributed as dist
import functools
from torch.testing._internal.common_utils import run_tests
from spmd.test._utils import DistTensorTestBase, with_comms  # type: ignore
from spmd import (
    distribute_module,
    distribute_tensor,
    DeviceMesh,
    DTensor,
    Shard,
    Replicate,
)


class SimpleModel(torch.nn.Module):
    def __init__(self, device):
        super(SimpleModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 16, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 12, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def _gradient_hook(param, grad):
    param._local_tensor.grad = grad._local_tensor


def shard_module(m, device_type):
    pg = dist.distributed_c10d._get_default_group()
    start_idx = 0
    device_mesh = DeviceMesh(
        device_type,
        list(range(start_idx, start_idx + pg.size())),
        dim_groups=[pg],
    )
    col_wise_sharding = [Shard(0)]
    row_wise_sharding = [Shard(1)]
    replicate = [Replicate()]

    def shard_params(name, module):
        if isinstance(module, nn.Linear):
            if name == "net1":
                sharded_weight = torch.nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, col_wise_sharding
                    )
                )
                sharded_bias = torch.nn.Parameter(
                    distribute_tensor(
                        module.bias, device_mesh, col_wise_sharding
                    )
                )
                module.register_parameter("weight", sharded_weight)
                module.register_parameter("bias", sharded_bias)
                module.weight.register_hook(
                    functools.partial(_gradient_hook, module.weight)
                )
            elif name == "net2":
                sharded_weight = torch.nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, row_wise_sharding
                    )
                )
                replicated_bias = torch.nn.Parameter(
                    distribute_tensor(module.bias, device_mesh, replicate)
                )
                module.register_parameter("weight", sharded_weight)
                module.register_parameter("bias", replicated_bias)

    def replicate_input(inputs):
        return DTensor.from_local(inputs[0], device_mesh, replicate)

    def aggregate_output(outputs):
        assert isinstance(outputs, DTensor)
        return (
            outputs.redistribute(outputs.device_mesh, replicate)
            .contiguous()
            .to_local()
        )

    dist_mod = distribute_module(
        m,
        device_mesh,
        partition_fn=shard_params,
        input_fn=replicate_input,
        output_fn=aggregate_output,
    )
    return dist_mod


class DistTensorMegatronTest(DistTensorTestBase):
    @with_comms
    def test_simple_megatron_e2e(self):
        LR = 0.5
        inp_size = [5, 10]
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        torch.manual_seed(5)
        model = SimpleModel(self.device_type)
        torch.manual_seed(5)
        model_tp = SimpleModel(self.device_type)
        model_tp = shard_module(model_tp, self.device_type)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()
        # This is for FSDP + TP integration.
        self.assertTrue(model_tp.net1.weight._local_tensor.grad is not None)

        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim.step()
        optim = torch.optim.SGD(model_tp.parameters(), lr=LR)
        optim.step()

        torch.manual_seed(3)
        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)


if __name__ == "__main__":
    run_tests()
