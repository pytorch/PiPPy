# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.nn as nn
from typing import Sequence, Tuple
from spmd.tensor import (
    distribute_tensor,
    DTensor,
    Shard,
    Replicate,
    DeviceMesh,
    Placement,
)
from spmd.tensor.parallel import (
    TensorParallelMultiheadAttention,
)


def replicate_input(
    inputs: Sequence[torch.Tensor], device_mesh: DeviceMesh
) -> Tuple[DTensor, ...]:
    replicate = [Replicate()] * device_mesh.ndim
    return tuple(
        DTensor.from_local(tensor, device_mesh, replicate) for tensor in inputs
    )


def replicate_output(output: DTensor, device_mesh: DeviceMesh) -> torch.Tensor:
    if isinstance(output, DTensor):
        replicate = [Replicate()] * output.device_mesh.ndim
        # TODO: can the output be left incontiguous?
        return (
            output.redistribute(output.device_mesh, replicate)
            .to_local()
            .contiguous()
        )


def tp_shard_self_attn(
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

def _parallelize_linear(
    module: nn.Module,
    parallel_style: ParallelStyle = ColwiseParallel(),
    device_mesh: DeviceMesh = None,
    tp_mesh_dim: int = 0,
) -> None:
    if not isinstance(module, nn.Linear):
        raise RuntimeError(
            f"{module} is not a torch.nn.Linear module but _parallelize_linear was called!"
        )

    if not isinstance(parallel_style, ParallelStyle):
        raise RuntimeError(
            f"parallel_style passed to _parallelize_linear is not a ParallelStyle object but {parallel_style}!"
        )

    col_wise_sharding: Sequence[Placement] = [Replicate()] * device_mesh.ndim
    col_wise_sharding[tp_mesh_dim] = Shard(0)
    row_wise_sharding: Sequence[Placement] = [Replicate()] * device_mesh.ndim
    row_wise_sharding[tp_mesh_dim] = Shard(1)
    replicate: Sequence[Placement] = [Replicate()] * device_mesh.ndim

    def linear_module_placements(
        parallel_style: ParallelStyle,
    ) -> Tuple[Sequence[Placement], Sequence[Placement]]:
        if isinstance(parallel_style, RowwiseParallel):
            return row_wise_sharding, replicate
        elif isinstance(parallel_style, ColwiseParallel):
            return col_wise_sharding, col_wise_sharding
        elif isinstance(parallel_style, PairwiseParallel):
            raise RuntimeError(f"{parallel_style} is not supported!")
        else:
            raise RuntimeError(f"{parallel_style} is not supported!")

    # placements
    linear_placements, bias_placements = linear_module_placements(
        parallel_style
    )
    # params
    weight = nn.Parameter(
        distribute_tensor(module.weight, device_mesh, linear_placements)
    )
    module.register_parameter("weight", weight)
    if module.bias is not None:
        bias = nn.Parameter(
            distribute_tensor(module.bias, device_mesh, bias_placements)
        )
        module.register_parameter("bias", bias)
    # input
    if parallel_style._prepare_input is not None:
        input_fn: Callable[
            [Sequence[Union[torch.Tensor, DTensor]], Tuple[DTensor, ...]]
        ] = lambda inputs: tuple(
            parallel_style._prepare_input(tensor, device_mesh)
            for tensor in inputs
        )
        module.register_forward_pre_hook(lambda _, inputs: input_fn(inputs))  # type: ignore
    # output
    if parallel_style._prepare_output is not None:
        output_fn = parallel_style._prepare_output
        module.register_forward_hook(
            lambda mod, inputs, outputs: output_fn(outputs)  # type: ignore
        )
