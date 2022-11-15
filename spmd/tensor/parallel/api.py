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


def _shard_input_1d(
    tensor: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
    dim: int = 0,
) -> DTensor:
    if device_mesh is None:
        if isinstance(DTensor, tensor):
            device_mesh = tensor.device_mesh

    assert device_mesh is not None, "_replicate_input_1d: device mesh is not 1D"

    sharding = [Shard(dim)]
    if isinstance(torch.Tensor, tensor):
        return DTensor.from_local(tensor, device_mesh, sharding)
    elif isinstance(DTensor, tensor):
        return tensor.redistribute(device_mesh, sharding)
    else:
        raise RuntimeError(
            "Input to _shard_input_1d must be either torch.Tensor or DTensor!"
        )


def MakeInputShard1D(dim: int) -> functools.partial[DTensor]:
    # This function generates input handler for 1-D mesh device only
    return functools.partial(_shard_input_1d, dim=dim)


def _replicate_input_1d(
    tensor: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
) -> DTensor:
    if device_mesh is None:
        if isinstance(DTensor, tensor):
            device_mesh = tensor.device_mesh

    assert device_mesh is not None, "_replicate_input_1d: device mesh is not 1D"

    replicate = [Replicate()]
    if isinstance(torch.Tensor, tensor):
        return DTensor.from_local(tensor, device_mesh, replicate)
    elif isinstance(DTensor, tensor):
        return tensor.redistribute(device_mesh, replicate)
    else:
        raise RuntimeError(
            "Input to _replicate_input_1d must be either torch.Tensor or DTensor!"
        )


def MakeInputReplicated() -> functools.partial[DTensor]:
    # This function generates input handler for 1-D mesh device only
    return functools.partial(_replicate_input_1d)


def _replicate_output_1d(tensor: DTensor) -> torch.Tensor:
    assert (
        isinstance(tensor, DTensor) and tensor.device_mesh.ndim == 1
    ), "_replicate_output_1d: device mesh is not 1D"
    return replicate_output(tensor)


def MakeOutputReplicated() -> functools.partial[torch.Tensor]:
    # This function generates input handler for 1-D mesh device only
    return functools.partial(_replicate_output_1d)


@dataclass
class ParallelStyle(object):
    """
    The parallel style user wants the module or submodule to be sharded.
    Users can extend this class to build their own parallel style with customized input/output preparations.
    """

    _prepare_input: Callable[
        [Union[torch.Tensor, DTensor], Optional[DeviceMesh]], DTensor
    ]
    _prepare_output: Callable[[DTensor], Union[Tensor, DTensor]]


class RowwiseParallel(ParallelStyle):
    """
    Partitioning the row of a module. We assume the input to be a Shard(-1) DTensor and output to be a replicated DTensor.
    """

    def __init__(self):
        super().__init__(MakeInputShard(-1), MakeOutputReplicated())


class ColwiseParallel(ParallelStyle):
    """
    Partitioning the column of a tensor or module. We assume the input to be a Replicated DTensor and output to be a Shard(-1) DTensor.
    """

    def __init__(self):
        super().__init__(MakeInputReplicated(), None)


class PairwiseParallel(ParallelStyle):
    """
    We concatenate colwise and rowwise styles as a fixed pair like what Megatron-LM(https://arxiv.org/abs/1909.08053) is doing. We assume both input and output to a Replicated DTensor. We now only support Multihead Attention, MLP and transformer for this style.
    We also need to assume the input is a nn.Multihead Attention, nn.Transformer or even-number layers of nn.Linear for now.
    """

    def __init__(self):
        super().__init__(MakeInputReplicated(), MakeOutputReplicated())


def _parallelize_linear_1d(
    module: nn.Module,
    parallel_style: ParallelStyle = ColwiseParallel(),
    device_mesh: DeviceMesh = None,
    tp_mesh_dim: int = 0,  # TODO: remove this param???
) -> None:
    if not isinstance(nn.Linear, module):
        raise RuntimeError(
            f"{module} is not a torch.nn.Linear module but _parallelize_linear was called!"
        )

    if not isinstance(ParallelStyle, parallel_style):
        raise RuntimeError(
            f"parallel_style passed to _parallelize_linear is not a ParallelStyle object but {parallel_style}!"
        )

    col_wise_sharding: Sequence[Placement] = [Shard(0)]
    row_wise_sharding: Sequence[Placement] = [Shard(1)]
    replicate: Sequence[Placement] = [Replicate()]

    def linear_module_placements(
        parallel_style: ParallelStyle,
    ) -> Tuple[Sequence[Placement], Sequence[Placement]]:
        if isinstance(RowwiseParallel, parallel_style):
            return row_wise_sharding, replicate
        elif isinstance(ColwiseParallel, parallel_style):
            return col_wise_sharding, col_wise_sharding
        elif isinstance(PairwiseParallel, parallel_style):
            raise RuntimeError(f"{parallel_style} is not supported!")
        else:
            raise RuntimeError(f"{parallel_style} is not supported!")

    # placements
    linear_placements, bias_placements = linear_module_placements(
        parallel_style
    )
    # params
    sharded_weight = nn.Parameter(
        distribute_tensor(module.weight, device_mesh, linear_placements)
    )
    module.register_parameter("weight", sharded_weight)
    if module.bias is not None:
        replicated_bias = nn.Parameter(
            distribute_tensor(module.bias, device_mesh, replicate)
        )
        module.register_parameter("bias", bias_placements)
    # input
    if parallel_style._prepare_input is not None:
        input_fn: Callable[
            [Sequence[Union[torch.Tensor, DTensor]], Tuple[DTensor, ...]]
        ] = lambda inputs: [
            parallel_style._prepare_input(tensor, device_mesh)
            for tensor in inputs
        ]
        module.register_forward_pre_hook(lambda _, inputs: input_fn(inputs))  # type: ignore
    # output
    if parallel_style._prepare_output is not None:
        output_fn = parallel_style._prepare_output
        module.register_forward_hook(
            lambda mod, inputs, outputs: output_fn(outputs)  # type: ignore
        )


# TODO: deprecate old TP api
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


def tp_shard_self_attn(
    device_mesh: DeviceMesh,
) -> Callable[[str, nn.Module], None]:
    return functools.partial(_shard_self_attn, device_mesh=device_mesh)


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
