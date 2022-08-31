# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Optional, Callable, Tuple, Union
import torch
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
            forward_pre_hook.
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. output_fn will be
            installed as a module forward_hook.

    Returns:
        A module that contains parameters/buffers that are all `DTensor`s.
    """

    if device_mesh is None:
        device_mesh = get_global_device_mesh()

    # set this to true on demand, to avoid inplace update the parameter directly
    # This is purely for nn.Module API parameter replacement BC reason
    overwrite_on_conversion = (
        torch.__future__.get_overwrite_module_params_on_conversion()
    )
    torch.__future__.set_overwrite_module_params_on_conversion(True)

    # this function loop over the whole module parameters
    # and buffers, replicate all non DTensor params/buffers
    # to DTensor parameters/buffers
    def replicate_module_params_buffers(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, torch.Tensor) and not isinstance(t, DTensor):
            # replicate the tensor if it has not been converted yet.
            assert device_mesh is not None
            return distribute_tensor(
                t, device_mesh, [Replicate()] * device_mesh.ndim
            )
        else:
            return t

    if partition_fn is None:
        # if partition_fn not specified, we by default replicate
        # all module params/buffers
        module._apply(replicate_module_params_buffers)
    else:
        # apply partition_fun to submodules
        for name, submod in module.named_modules():
            partition_fn(name, submod)
            # replicate the rest of params/buffers if not been partitioned
            # in the partition_fn, we can't easily use `module._apply` again
            # here because we don't know what happened inside partition_fn
            # as user could do anything, i.e. install hooks, and we want
            # to preserve those.
            for key, param in submod._parameters.items():
                if not isinstance(param, DTensor):
                    submod.register_parameter(
                        key,
                        nn.Parameter(replicate_module_params_buffers(param)),
                    )
            for key, buffer in submod._buffers.items():
                if not isinstance(buffer, DTensor):
                    submod._buffers[key] = replicate_module_params_buffers(
                        param
                    )

    # register input_fn as module forward pre hook
    if input_fn is not None:
        module.register_forward_pre_hook(lambda _, inputs: input_fn(inputs))  # type: ignore
    # register input_fn as module forward hook
    if output_fn is not None:
        module.register_forward_hook(
            lambda mod, inputs, outputs: output_fn(outputs)  # type: ignore
        )

    # restore the overwrite_on_conversion state
    torch.__future__.set_overwrite_module_params_on_conversion(
        overwrite_on_conversion
    )
    return module
