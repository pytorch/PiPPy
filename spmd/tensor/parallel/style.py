# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from typing import Optional
from spmd.tensor import DTensor, Shard, Replicate, DeviceMesh
from spmd.tensor.parallel.utils import _prepare_output_validate


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
        (DTensor): A DTensor made repliacted.
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
