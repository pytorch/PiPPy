# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
import functools
from typing import Sequence, Tuple, Callable, cast
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


def _is_mlp(module: nn.Module) -> bool:
    # We assume that the structure of MLP to shard defined in object way as below:
    #   * -> (Linear -> * -> Linear)+ -> *
    # positive even number of Linear layers interleaved by activation/norm/dropout
    linear_submodules = list(
        filter(lambda x: isinstance(x, nn.Linear), module.children())
    )
    return len(linear_submodules) > 0 and len(linear_submodules) % 2 == 0


def _shard_mlp(name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
    if _is_mlp(module):
        col_wise_sharding: Sequence[Placement] = [Shard(0)]
        row_wise_sharding: Sequence[Placement] = [Shard(1)]
        replicate: Sequence[Placement] = [Replicate()] * device_mesh.ndim
        linear_submodules = list(
            filter(lambda x: isinstance(x, nn.Linear), module.children())
        )

        for i, m in enumerate(linear_submodules):
            if i % 2 == 0:
                # shard Linear layer column-wisely
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        cast(torch.Tensor, m.weight),
                        device_mesh,
                        col_wise_sharding,
                    )
                )
                sharded_bias = nn.Parameter(
                    distribute_tensor(
                        cast(torch.Tensor, m.bias),
                        device_mesh,
                        col_wise_sharding,
                    )
                )
                m.register_parameter("weight", sharded_weight)
                m.register_parameter("bias", sharded_bias)
            else:
                # shard Linear layer row-wisely
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        cast(torch.Tensor, m.weight),
                        device_mesh,
                        row_wise_sharding,
                    )
                )
                replicated_bias = nn.Parameter(
                    distribute_tensor(
                        cast(torch.Tensor, m.bias), device_mesh, replicate
                    )
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
