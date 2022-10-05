# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
import functools
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import DistTensorTestBase, with_comms, NUM_DEVICES, skip_unless_torch_gpu  # type: ignore
from spmd import (
    distribute_tensor,
    distribute_module,
    DeviceMesh,
    DTensor,
    Shard,
    Replicate,
)
from spmd.tensor.parallel import TensorParallelMultiheadAttention, shard_self_attn, replicate_input


class MLPModule(torch.nn.Module):
    def __init__(self, device):
        super(MLPModule, self).__init__()
        torch.manual_seed(5)
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


def _gradient_hook(param, grad):
    param._local_tensor.grad = grad._local_tensor


def shard_mlp(m, device_type, tp_size):
    start_idx = 0
    device_mesh = DeviceMesh(
        device_type,
        list(range(start_idx, start_idx + tp_size)),
    )
    col_wise_sharding = [Shard(0)]
    row_wise_sharding = [Shard(1)]
    replicate = [Replicate()]

    def shard_params(name, module):
        if isinstance(module, nn.Linear):
            if name == "net1":
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, col_wise_sharding
                    )
                )
                sharded_bias = nn.Parameter(
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
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, row_wise_sharding
                    )
                )
                replicated_bias = nn.Parameter(
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


class MultiheadAttnWrap(nn.Module):
    # TODO: complete the interface
    def __init__(self, embed_dim, num_heads, add_bias_kv=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, add_bias_kv=add_bias_kv)

    def forward(self, query, key, value):
        return self.attn(query, key, value)

class DistTensorParallelExampleTest(DistTensorTestBase):
    @with_comms
    def test_mlp_megatron_e2e(self):
        inp_size = [5, 10]
        # Ensure all tp ranks have same input.
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)
        model_tp = MLPModule(self.device_type)

        # Ensure model are initialized the same way.
        self.assertEqual(model.net1.weight, model_tp.net1.weight)
        self.assertEqual(model.net1.bias, model_tp.net1.bias)
        self.assertEqual(model.net2.weight, model_tp.net2.weight)
        self.assertEqual(model.net2.bias, model_tp.net2.bias)

        # Shard module and initialize optimizer.
        LR = 0.25
        shard_mlp(model_tp, self.device_type, NUM_DEVICES)
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()
        # This is for FSDP + TP integration.
        self.assertTrue(model_tp.net1.weight._local_tensor.grad is not None)

        replicate = [Replicate()]
        device_mesh = model_tp.net1.weight.device_mesh

        # Ensure gradients are same.
        self.assertEqual(
            model.net1.weight.grad,
            model_tp.net1.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net1.bias.grad,
            model_tp.net1.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net2.weight.grad,
            model_tp.net2.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net2.bias.grad,
            model_tp.net2.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        self.assertEqual(
            model.net1.weight,
            model_tp.net1.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net1.bias,
            model_tp.net1.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.net2.weight,
            model_tp.net2.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        # Due to the trick we use for Partial aggregation, we only check the weight when local_rank = 0.
        if self.rank == 0:
            self.assertEqual(
                model.net2.bias,
                model_tp.net2.bias.redistribute(
                    device_mesh=device_mesh, placements=replicate
                ).to_local(),
            )

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)

    # baddbmm introduces nan occasionally on CPU: https://github.com/pytorch/pytorch/issues/80588
    @with_comms
    #@skip_unless_torch_gpu
    def test_self_attn_megatron_e2e(self):
        inp_size = [8, 12, 16]
        # Ensure all tp ranks have same input.
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)

        # Initialize model using same seed.
        torch.manual_seed(5)
        model = TensorParallelMultiheadAttention(
            16, 8, self.device_type, tp_size=4, add_bias_kv=True
        )
        torch.manual_seed(5)
        model_tp = TensorParallelMultiheadAttention(
            16, 8, self.device_type, tp_size=4, add_bias_kv=True
        )

        # Ensure model are initialized the same way.
        self.assertEqual(model.qkv.weight, model_tp.qkv.weight)
        self.assertEqual(model.qkv.bias, model_tp.qkv.bias)
        self.assertEqual(model.proj.weight, model_tp.proj.weight)
        self.assertEqual(model.proj.bias, model_tp.proj.bias)

        # Shard module and initialize optimizer.
        # TODO BE: device_mesh will be instantiated twice.
        device_mesh = DeviceMesh(self.device_type, NUM_DEVICES)
        distribute_module(model_tp, device_mesh, partition_fn=shard_self_attn(self.device_type, NUM_DEVICES), input_fn=replicate_input(device_mesh), output_fn=None)

        LR = 0.25
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp, inp, inp)
        output_tp = model_tp(inp, inp, inp)
        self.assertEqual(output, output_tp)

        output.sum().backward()
        output_tp.sum().backward()

        replicate = [Replicate()]
        device_mesh = model_tp.qkv.weight.device_mesh
        # Ensure gradients are same.
        self.assertEqual(
            model.qkv.weight.grad,
            model_tp.qkv.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias.grad,
            model_tp.qkv.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight.grad,
            model_tp.proj.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias.grad,
            model_tp.proj.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        self.assertEqual(
            model.qkv.weight,
            model_tp.qkv.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias,
            model_tp.qkv.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight,
            model_tp.proj.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias,
            model_tp.proj.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp, inp, inp)
        output_tp = model_tp(inp, inp, inp)
        self.assertEqual(output, output_tp)

"""
    # note: test our new shard_self_attn api
    # baddbmm introduces nan occasionally on CPU: https://github.com/pytorch/pytorch/issues/80588
    @with_comms
    def test_self_attn_megatron_e2e_2(self):
        device_mesh = DeviceMesh(self.device_type, 4)
        inp_size = [8, 12, 16]
        # Ensure all tp ranks have same input.
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)

        # Initialize model using same seed.
        torch.manual_seed(5)
        model = nn.MultiheadAttention(16, 8, add_bias_kv=True)
        torch.manual_seed(5)
        # TODO: our sharding function cannot shard the root node
        model_tp = MultiheadAttnWrap(16, 8, add_bias_kv=True)

        # TODO: input/output fn
        distribute_module(model_tp, device_mesh, partition_fn=my_shard_self_attn(self.device_type, 4), input_fn=None, output_fn=None)

        replicate = [Replicate()]
        device_mesh = model_tp.attn.qkv.weight.device_mesh
        # Ensure model are initialized the same way.
        self.assertEqual(
            model.in_proj_weight,
            model_tp.attn.qkv.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.in_proj_bias,
            model_tp.attn.qkv.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.out_proj.weight,
            model_tp.attn.proj.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.out_proj.bias,
            model_tp.attn.proj.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        LR = 0.25
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp, inp, inp)[0]
        output_tp = model_tp(inp, inp, inp)
        if self.rank == 0:
            print(f"output={output}\noutput_tp={output_tp}")
        #print(f"output shape={output.shape}, output_tp shape={output_tp.shape}")
        self.assertEqual(output, output_tp)  # FIX: ValueError: tensors are not close

        output.sum().backward()
        output_tp.sum().backward()

        replicate = [Replicate()]
        device_mesh = model_tp.attn.qkv.weight.device_mesh
        # Ensure gradients are same.
        self.assertEqual(
            model.in_proj_weight.grad,
            model_tp.attn.qkv.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.in_proj_bias.grad,
            model_tp.attn.qkv.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.out_proj.weight.grad,
            model_tp.attn.proj.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.out_proj.bias.grad,
            model_tp.attn.proj.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        optim.step()
        optim_tp.step()

        # Ensure model weights are still same after update.
        self.assertEqual(
            model.qkv.weight,
            model_tp.attn.qkv.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.qkv.bias,
            model_tp.attn.qkv.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.weight,
            model_tp.attn.proj.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.proj.bias,
            model_tp.attn.proj.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp, inp, inp)
        output_tp = model_tp(inp, inp, inp)
        self.assertEqual(output, output_tp)
"""

if __name__ == "__main__":
    run_tests()
