# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
import functools
from spmd import (
    distribute_tensor,
    distribute_module,
    DeviceMesh,
    DTensor,
    Shard,
    Replicate,
)
from spmd.tensor.parallel import TensorParallelMultiheadAttention


def _replicate_input(inputs, device_mesh):
    DTensors = []
    for tensor in inputs:
        DTensors.append(DTensor.from_local(tensor, device_mesh, [Replicate()]))
    return tuple(DTensors)


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


def _shard_self_attn(name, module, device_type, tp_size) -> None:
    # note: do we really need start_idx since it's 0???
    start_idx = 0
    device_mesh = DeviceMesh(
        device_type,
        list(range(start_idx, start_idx + tp_size)),
    )
    col_wise_sharding = [Shard(0)]
    row_wise_sharding = [Shard(1)]
    replicate = [Replicate()]

    def _shard_self_attn_params(name, module):
        if isinstance(module, nn.Linear):
            if name == "qkv":
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, col_wise_sharding
                    )
                )
                module.register_parameter("weight", sharded_weight)
                module.weight.register_hook(
                    functools.partial(_gradient_hook, module.weight)
                )
                if module.bias is not None:
                    sharded_bias = nn.Parameter(
                        distribute_tensor(
                            module.bias, device_mesh, col_wise_sharding
                        )
                    )
                    module.register_parameter("bias", sharded_bias)
                    module.bias.register_hook(
                        functools.partial(_gradient_hook, module.bias)
                    )
            elif name == "proj":
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, row_wise_sharding
                    )
                )
                module.register_parameter("weight", sharded_weight)
                module.weight.register_hook(
                    functools.partial(_gradient_hook, module.weight)
                )
                _aggregate_local_tensor(module)
                if module.bias is not None:
                    replicated_bias = nn.Parameter(
                        distribute_tensor(module.bias, device_mesh, replicate)
                    )
                    module.register_parameter("bias", replicated_bias)
                    module.bias.register_hook(
                        functools.partial(_gradient_hook, module.bias)
                    )

    if isinstance(module, TensorParallelMultiheadAttention):
        for n, m in module.named_children():
            _shard_self_attn_params(n, m)


def _replace_with_custom_self_attn(name, module, device_type, tp_size) -> None:
    # named_modules() produces a prefix iterator over the tree
    # for each module in named_modules(), we check if there's any MultiheadAttention module in its immediate children
    # if any, replace it with TensorParallelMultiheadAttention using register_module() and shard
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            tp_multi_head_attention = TensorParallelMultiheadAttention(
                child.embed_dim,
                child.num_heads,
                device=device_type,
                tp_size=tp_size,
                add_bias_kv=True,  # TODO: can we recover this info from child???
            )
            # TODO: copy parameters. Can we merge the construction step with the sharding step?
            tp_multi_head_attention.copy(child)
            #_shard_custom_multi_head_attn(tp_multi_head_attention, device_type, tp_size)
            module.register_module(name, tp_multi_head_attention)


def shard_self_attn(device_type, tp_size):
    return functools.partial(
        _shard_self_attn, device_type=device_type, tp_size=tp_size
    )

def replicate_input(device_mesh):
    return functools.partial(
        _replicate_input, device_mesh=device_mesh
    )
