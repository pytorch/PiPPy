# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from spmd.test._utils import DistTensorTestBase, with_comms
from spmd import (
    distribute_tensor,
    distribute_module,
    DeviceMesh,
    DTensor,
    Shard,
    Replicate,
)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 12)

    def forward(self, x):
        return self.net4(self.gelu(self.net3(self.net2(self.relu(self.net1(x))))))


def _aggregate_local_tensor(module: torch.nn.Module) -> torch.nn.Module:
    def hook_func(_module, _input, output):
        if isinstance(output, Tensor):
            replica_placement = [Replicate()]
            return (
                output.redistribute(output.device_mesh, replica_placement)
                .contiguous()
                .local_tensor()
            )

    module.register_forward_hook(hook_func)
    return module


def _replicate_input_tensor(
    module: torch.nn.Module, device_mesh, replica_placement
) -> torch.nn.Module:
    def hook_func(_, input):
        if not isinstance(input[0], Tensor):
            return Tensor.from_local(input[0], device_mesh, replica_placement)

    module.register_forward_pre_hook(hook_func)
    return module


def _gradient_hook(param, grad):
    param._local_tensor.grad = grad._local_tensor


def shard_module(m):
    pg = dist.distributed_c10d._get_default_group()
    start_idx = 0
    device_mesh = DeviceMesh(
        "cuda", list(range(start_idx, start_idx + pg.size())), dim_groups=[pg]
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
    m.net1.weight.register_hook(functools.partial(_gradient_hook, m.net1.weight))


class DistTensorMegatronTest(DistTensorTestBase):
    @with_comms
    def test_simple_megatron_e2e(self):
        LR = 0.5
        inp_size = [5, 10]
        torch.manual_seed(0)
        inp = torch.rand(*inp_size).cuda(self.rank)
        torch.manual_seed(5)
        model = SimpleModel().cuda(self.rank)
        torch.manual_seed(5)
        model_tp = SimpleModel().cuda(self.rank)
        shard_module(model_tp)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()
        # self.assertTrue(model_tp.net1.weight.local_tensor().grad is not None)

        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim.step()
        optim = torch.optim.SGD(model_tp.parameters(), lr=LR)
        optim.step()

        torch.manual_seed(3)
        inp = torch.rand(*inp_size).cuda(self.rank)
        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)


if __name__ == "__main__":
    run_tests()
