# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.distributed as dist
import functools
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd import distribute_tensor, DeviceMesh, DTensor, Shard, Replicate


class SimpleModel(torch.nn.Module):
    def __init__(self, device):
        super(SimpleModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 16, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 12, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def _aggregate_local_tensor(module: torch.nn.Module) -> torch.nn.Module:
    def hook_func(_module, _input, output):
        if isinstance(output, DTensor):
            replica_placement = [Replicate()]
            return (
                output.redistribute(output.device_mesh, replica_placement)
                .contiguous()
                .to_local()
            )

    module.register_forward_hook(hook_func)
    return module


def _replicate_input_tensor(
    module: torch.nn.Module, device_mesh, replica_placement
) -> torch.nn.Module:
    def hook_func(_, input):
        if not isinstance(input[0], DTensor):
            return DTensor(input[0], device_mesh, replica_placement)

    module.register_forward_pre_hook(hook_func)
    return module


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
    m.net1.weight = torch.nn.Parameter(
        distribute_tensor(m.net1.weight, device_mesh, col_wise_sharding)
    )
    m.net2.weight = torch.nn.Parameter(
        distribute_tensor(m.net2.weight, device_mesh, row_wise_sharding)
    )
    m.net1.bias = torch.nn.Parameter(
        distribute_tensor(m.net1.bias, device_mesh, col_wise_sharding)
    )
    m.net2.bias = torch.nn.Parameter(
        distribute_tensor(m.net2.bias, device_mesh, replicate)
    )
    m = _replicate_input_tensor(m, device_mesh, replicate)
    m.net2 = _aggregate_local_tensor(m.net2)
    m.net1.weight.register_hook(
        functools.partial(_gradient_hook, m.net1.weight)
    )


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
        shard_module(model_tp, self.device_type)

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
