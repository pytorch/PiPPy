# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from dataclasses import dataclass
from abc import ABC
from typing import Callable, Union, Optional
from spmd.tensor import (
    DTensor,
    Shard,
    Replicate,
    DeviceMesh,
)


@dataclass
class ParallelStyle(ABC):
    """
    The parallel style user wants the module or submodule to be parallelized.
    Users can extend this class to build their own parallel style with customized input/output preparations.
    """

    _prepare_input: Optional[
        Callable[
            [Union[torch.Tensor, DTensor], Optional[DeviceMesh], Optional[int]],
            DTensor,
        ]
    ]
    _prepare_output: Optional[
        Callable[
            [DTensor, Optional[DeviceMesh], Optional[int]],
            Union[torch.Tensor, DTensor],
        ]
    ]


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
    if device_mesh is None:
        if isinstance(input, DTensor):
            device_mesh = input.device_mesh

    assert (
        device_mesh is not None
    ), "device_mesh is not passed nor can be inferred"
    assert (
        device_mesh.ndim == 1
    ), f"device_mesh dim is {device_mesh.ndim} but expcted to be 1"

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
    if device_mesh is None:
        if isinstance(input, DTensor):
            device_mesh = input.device_mesh

    assert (
        device_mesh is not None
    ), "device_mesh is not passed nor can be inferred"
    assert (
        device_mesh.ndim == 1
    ), f"device_mesh dim is {device_mesh.ndim} but expcted to be 1"

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
