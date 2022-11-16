# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from dataclasses import dataclass
from typing import Callable, Union, Optional
from spmd.tensor import (
    DTensor,
    Shard,
    Replicate,
    DeviceMesh,
)


@dataclass
class ParallelStyle(object):
    """
    The parallel style user wants the module or submodule to be sharded.
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


def shard_input_1d(
    tensor: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
    dim: int = 0,
) -> DTensor:
    # This function shards input tensor on `dim` over an 1-D device_mesh
    if device_mesh is None:
        if isinstance(tensor, DTensor):
            device_mesh = tensor.device_mesh

    assert (
        device_mesh is not None
    ), "device_mesh is not passed nor can be inferred in shard_input_1d"
    assert device_mesh.ndim == 1, "shard_input_1d: device mesh is not 1D"

    shard_spec = [Shard(dim)]
    if isinstance(tensor, DTensor):
        return tensor.redistribute(device_mesh, shard_spec)
    elif isinstance(tensor, torch.Tensor):
        return DTensor.from_local(tensor, device_mesh, shard_spec)
    else:
        raise RuntimeError(
            "Input to shard_input_1d must be either torch.Tensor or DTensor!"
        )


def replicate_input_1d(
    tensor: Union[torch.Tensor, DTensor],
    device_mesh: Optional[DeviceMesh] = None,
) -> DTensor:
    # This function replicates input tensor over an 1-D device_mesh
    if device_mesh is None:
        if isinstance(tensor, DTensor):
            device_mesh = tensor.device_mesh

    assert (
        device_mesh is not None
    ), "device_mesh is not passed nor can be inferred in replicate_input_1d"
    assert device_mesh.ndim == 1, "replicate_input_1d: device mesh is not 1D"

    replicate = [Replicate()]
    if isinstance(tensor, DTensor):
        return tensor.redistribute(device_mesh, replicate)
    elif isinstance(tensor, torch.Tensor):
        return DTensor.from_local(tensor, device_mesh, replicate)
    else:
        raise RuntimeError(
            "Input to replicate_input_1d must be either torch.Tensor or DTensor!"
        )
