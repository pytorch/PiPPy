# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Optional, Callable
import torch.nn as nn
from spmd.tensor import distribute_tensor
from spmd.tensor.api import DTensor
from spmd.tensor.device_mesh import get_global_device_mesh, DeviceMesh
from spmd.tensor.placement_types import Replicate, Shard


def distribute_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, nn.Module], None]] = None,
    input_fn: Optional[Callable[..., None]] = None,
    output_fn: Optional[Callable[..., None]] = None,
) -> nn.Module:
    """
    This function converts all module parameters to :class:`DTensor` parameters
    according to the `partition_fn` specified. It could also control the input or
    output of the module by specifying the `input_fn` and `output_fn`. (i.e. convert
    the input to :class:`DTensor`, convert the output back to torch.Tensor)
    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the `device_mesh`). If `partition_fn` is not specified,
            by default we replicate all module parameters of `module` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. `input_fn` will be installed as a module
            `forward_pre_hook` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. output_fn will be
            installed as a module `forward_hook` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all `DTensor`s.
    """

    if device_mesh is None:
        device_mesh = get_global_device_mesh()

    def replicate_module_params_buffers(m: nn.Module, mesh: DeviceMesh) -> None:
        # This function loop over the immediate module parameters and
        # buffers, replicate all non DTensor params/buffers to DTensor
        # parameters/buffers, if they have not been partitioned in the
        # partition_fn, we can't easily use `module._apply` here
        # because we don't know what happened inside partition_fn as
        # user could do anything, i.e. install hooks, and we want to
        # preserve those.
        full_replicate = [Replicate()] * mesh.ndim
        for key, param in m._parameters.items():
            if param is not None and not isinstance(param, DTensor):
                m.register_parameter(
                    key,
                    nn.Parameter(
                        distribute_tensor(param.data, mesh, full_replicate)
                    ),
                )
        for key, buffer in m._buffers.items():
            if buffer is not None and not isinstance(buffer, DTensor):
                m._buffers[key] = distribute_tensor(
                    buffer, mesh, full_replicate
                )

    if partition_fn is None:
        # if partition_fn not specified, we by default replicate
        # all module params/buffers
        for name, submod in module.named_modules():
            replicate_module_params_buffers(submod, device_mesh)
    else:
        # apply partition_fun to submodules
        for name, submod in module.named_modules():
            partition_fn(name, submod)
            replicate_module_params_buffers(submod, device_mesh)

    # register input_fn as module forward pre hook
    if input_fn is not None:
        module.register_forward_pre_hook(lambda _, inputs: input_fn(inputs))  # type: ignore
    # register input_fn as module forward hook
    if output_fn is not None:
        module.register_forward_hook(
            lambda mod, inputs, outputs: output_fn(outputs)  # type: ignore
        )

    return module


# All public APIs from spmd package
__all__ = [
    "DTensor",
    "DeviceMesh",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
]
