# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from dataclasses import dataclass
from abc import ABC
from typing import Union, Optional
from spmd.tensor import (
    DTensor,
    Shard,
    Replicate,
    DeviceMesh,
)
from spmd.tensor.parallel.utils import (
    _Prepare_Input_Func_Type,
    _Prepare_Output_Func_Type,
    _prepare_input_validate,
    _prepare_output_validate,
)


@dataclass
class ParallelStyle(ABC):
    """
    The parallel style user wants the module or submodule to be parallelized.
    Users can extend this class to build their own parallel style with customized input/output preparations.
    """

    _prepare_input: Optional[_Prepare_Input_Func_Type]
    _prepare_output: Optional[_Prepare_Output_Func_Type]


@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_shard_1d(
    input: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
    dim: int = 0,
) -> DTensor:
    """
    Shard input tensor on `dim` over an 1-D device mesh. This function will be used in ParallelStyle.

    Args:
        input (Union[Tensor, DTensor]):
            This single tensor will be sharded on dimension `dim`
            over the 1-D :class:`DeviceMesh`.
        device_mesh (DeviceMesh, optional):
            The 1-D device mesh where `input` will be sharded.
            If no :class:`DeviceMesh` is passed and `input` is a :class:`DTensor`,
            `input.device_mesh` will be used.
            If :class:`DeviceMesh` is not 1-D, an exception will be thrown.
            Default: ``None``
        dim (int, optional): The sharding dimension of `input` tensor.
            Default: 0

    Returns:
        A :class:`DTensor` sharded on dimension `dim` over `device_mesh`.
    """
    shard_spec = [Shard(dim)]
    if isinstance(input, DTensor):
        return input.redistribute(device_mesh, shard_spec)
    elif isinstance(input, torch.Tensor):
        return DTensor.from_local(
            input, device_mesh, shard_spec, run_check=False
        )
    else:
        raise RuntimeError(
            f"Tensor parallel module expects torch.Tensor or DTensor input but received {type(input)}!"
        )


@_prepare_input_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_input_replicate_1d(
    input: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
) -> DTensor:
    """
    Replicate input tensor over an 1-D device mesh. This function will be used in ParallelStyle.

    Args:
        input (Union[Tensor, DTensor]):
            This single tensor will be replicated over the 1-D :class:`DeviceMesh`.
        device_mesh (DeviceMesh, optional):
            The 1-D device mesh where `input` will be replicated.
            If no :class:`DeviceMesh` is passed and `input` is a :class:`DTensor`,
            `input.device_mesh` will be used.
            If :class:`DeviceMesh` is not 1-D, an exception will be thrown.
            Default: ``None``

    Returns:
        A :class:`DTensor` replicated over `device_mesh`.
    """
    replicate = [Replicate()]
    if isinstance(input, DTensor):
        return input.redistribute(device_mesh, replicate)
    elif isinstance(input, torch.Tensor):
        return DTensor.from_local(
            input, device_mesh, replicate, run_check=False
        )
    else:
        raise RuntimeError(
            f"Tensor parallel module expects torch.Tensor or DTensor input but received {type(input)}!"
        )


@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_shard_1d(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None, dim: int = 0
) -> DTensor:
    """
    Convert Output DTensor to a sharded DTensor. This will be used in ParallelStyle.
    Args:
        output (DTensor): output of module to be converted.
        device_mesh (Optional[DeviceMesh]): DeviceMesh to shard the output DTensor.
            This needs to be a 1D DeviceMesh and we will throw exceptions if a
            non-1D DeviceMesh is passed in. If no DeviceMesh is passed in, we will
            reuse the one from output DTensor.
        dim (int): Sharding dim for output DTensor.
    Return:
        (DTensor): A DTensor sharded on the given dim.
    """

    return output.redistribute(device_mesh, [Shard(dim)])


@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_replicate_1d(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None
) -> DTensor:
    """
    Convert Output DTensor to a replicated DTensor. This will be used in ParallelStyle.
    Args:
        output (DTensor): output of module to be converted.
        device_mesh (Optional[DeviceMesh]): DeviceMesh to replicate the output DTensor.
            This needs to be a 1D DeviceMesh and we will throw exceptions if a non-1D
            DeviceMesh is passed in. If no DeviceMesh is passed in, we will reuse the
            one from output DTensor.
    Return:
        (DTensor): A DTensor made replicate.
    """

    return output.redistribute(device_mesh, [Replicate()])


@_prepare_output_validate  # type: ignore[arg-type] # pyre-ignore[56]
def make_output_tensor(
    output: DTensor, device_mesh: Optional[DeviceMesh] = None
) -> torch.Tensor:
    """
    Convert Output DTensor to a replicated DTensor first and then convert it to Tensor.
    Args:
        output (DTensor): output of module to be converted.
        device_mesh (Optional[DeviceMesh]): DeviceMesh to replicate the output DTensor.
            This needs to be a 1D DeviceMesh and we will throw exceptions if a non-1D
            DeviceMesh is passed in. If no DeviceMesh is passed in, we will reuse the
            one from output DTensor.
    Return:
        (torch.Tensor): A tensor converted from output DTensor.
    """

    return make_output_replicate_1d(  # type: ignore[attr-defined]
        output, device_mesh
    ).to_local()  # type: ignore[call-arg]
