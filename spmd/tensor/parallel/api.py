# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
import functools
from typing import Sequence, Tuple, Callable
from spmd.tensor import (
    distribute_tensor,
    DTensor,
    Shard,
    Replicate,
    DeviceMesh,
    Placement,
)
from spmd.tensor.parallel import TensorParallelMultiheadAttention


def _replicate_input(
    inputs: Sequence[torch.Tensor], device_mesh: DeviceMesh
) -> Tuple[DTensor, ...]:
    replicate = [Replicate()] * device_mesh.ndim
    return tuple(
        DTensor.from_local(tensor, device_mesh, replicate) for tensor in inputs
    )


def _shard_self_attn(
    name: str, module: nn.Module, device_mesh: DeviceMesh
) -> None:
    col_wise_sharding: Sequence[Placement] = [Shard(0)]
    row_wise_sharding: Sequence[Placement] = [Shard(1)]
    replicate: Sequence[Placement] = [Replicate()] * device_mesh.ndim

    def _shard_self_attn_params(name: str, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if name == "qkv":
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, col_wise_sharding
                    )
                )
                module.register_parameter("weight", sharded_weight)
                if module.bias is not None:
                    sharded_bias = nn.Parameter(
                        distribute_tensor(
                            module.bias, device_mesh, col_wise_sharding
                        )
                    )
                    module.register_parameter("bias", sharded_bias)
            elif name == "proj":
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, row_wise_sharding
                    )
                )
                module.register_parameter("weight", sharded_weight)
                if module.bias is not None:
                    replicated_bias = nn.Parameter(
                        distribute_tensor(module.bias, device_mesh, replicate)
                    )
                    module.register_parameter("bias", replicated_bias)

    if isinstance(module, TensorParallelMultiheadAttention):  # shard TPMA
        for n, m in module.named_children():
            _shard_self_attn_params(n, m)
    else:
        for n, m in module.named_children():  # replace with TPMA
            if isinstance(m, nn.MultiheadAttention):
                tp_multi_head_attention = TensorParallelMultiheadAttention(
                    m.embed_dim,
                    m.num_heads,
                    device=torch.device(device_mesh.device_type),
                    tp_size=device_mesh.size(0),  # group size on dim 0
                    add_bias_kv=m.bias_k is not None,
                )
                tp_multi_head_attention.copy(m)
                module.register_module(n, tp_multi_head_attention)


def _isMLP(module: nn.Module) -> bool:
    # We assume that the structure of MLP to shard defined in object way as below:
    # Linear -> ReLU -> Linear
    submodules = list(module.children())
    return (
        len(submodules) == 3
        and isinstance(submodules[0], nn.Linear)
        and isinstance(submodules[1], nn.ReLU)
        and isinstance(submodules[2], nn.Linear)
    )


def _gradient_hook(param: DTensor, grad: DTensor) -> None:
    param._local_tensor.grad = grad._local_tensor


def _shard_mlp(name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
    if _isMLP(module):
        col_wise_sharding: Sequence[Placement] = [Shard(0)]
        row_wise_sharding: Sequence[Placement] = [Shard(1)]
        replicate: Sequence[Placement] = [Replicate()] * device_mesh.ndim
        submodules = list(module.children())

        # shard Linear layer 1
        m = submodules[0]
        if isinstance(m, nn.Linear):
            sharded_weight = nn.Parameter(
                distribute_tensor(m.weight, device_mesh, col_wise_sharding)
            )
            sharded_bias = nn.Parameter(
                distribute_tensor(m.bias, device_mesh, col_wise_sharding)
            )
            m.register_parameter("weight", sharded_weight)
            m.register_parameter("bias", sharded_bias)
            # note: this hook enables access to m.weight._local_tensor.grad
            m.weight.register_hook(functools.partial(_gradient_hook, m.weight))

        # shard Linear layer 2
        m = submodules[2]
        if isinstance(m, nn.Linear):
            sharded_weight = nn.Parameter(
                distribute_tensor(m.weight, device_mesh, row_wise_sharding)
            )
            replicated_bias = nn.Parameter(
                distribute_tensor(m.bias, device_mesh, replicate)
            )
            m.register_parameter("weight", sharded_weight)
            m.register_parameter("bias", replicated_bias)


# Public APIs
def tp_shard_self_attn(
    device_mesh: DeviceMesh,
) -> Callable[[str, nn.Module], None]:
    return functools.partial(_shard_self_attn, device_mesh=device_mesh)


def tp_shard_mlp(
    device_mesh: DeviceMesh,
) -> Callable[[str, nn.Module], None]:
    return functools.partial(_shard_mlp, device_mesh=device_mesh)


def replicate_input(
    device_mesh: DeviceMesh,
) -> functools.partial[Tuple[DTensor, ...]]:
    return functools.partial(_replicate_input, device_mesh=device_mesh)


def replicate_output(output: DTensor) -> torch.Tensor:
    if isinstance(output, DTensor):
        replicate = [Replicate()] * output.device_mesh.ndim
        # TODO: can the output be left incontiguous?
        return (
            output.redistribute(output.device_mesh, replicate)
            .to_local()
            .contiguous()
        )
