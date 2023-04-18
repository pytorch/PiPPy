import os
import json
import gc
from typing import Optional, Union

import torch
from torch import nn


def load_checkpoint(
    model: nn.Module,
    index_filename: Union[str, os.PathLike],
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    checkpoint_folder = os.path.split(index_filename)[0]
    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    if "weight_map" in index:
        index = index["weight_map"]
    checkpoint_files = sorted(list(set(index.values())))
    checkpoint_files = [
        os.path.join(checkpoint_folder, f) for f in checkpoint_files
    ]
    for checkpoint_file in checkpoint_files:
        checkpoint = torch.load(checkpoint_file)
        for param_name, param in checkpoint.items():
            # Some weights like word_embeddings.weight and shared.weight will be used in different layers, but these layers
            # may not in the index file, so we can only clone the shared weight to their corresponding layers.
            if param_name in [
                "word_embeddings.weight",
                "shared.weight",
                "wte.weight",
                "decoder.embed_tokens.weight",
                "encoder.embed_tokens.weight",
            ]:
                if hasattr(model, "lm_head"):
                    model.lm_head.weight = torch.nn.Parameter(  # type: ignore[union-attr]
                        (param.clone()).to(device).to(dtype)
                    )
                if hasattr(model, "encoder_embed_tokens"):
                    model.encoder_embed_tokens.weight = torch.nn.Parameter(  # type: ignore[union-attr]
                        (param.clone()).to(device).to(dtype)
                    )
                if hasattr(model, "decoder_embed_tokens"):
                    model.decoder_embed_tokens.weight = torch.nn.Parameter(  # type: ignore[union-attr]
                        (param.clone()).to(device).to(dtype)
                    )
            set_module_tensor_to_device(
                model, param_name, device, value=param, dtype=dtype
            )
        del checkpoint
        gc.collect()

    return model


def set_module_tensor_to_device(
    module: nn.Module,
    param_name: str,
    device: Optional[torch.device] = None,
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).
    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        param_name (`str`):
            The full name of the parameter/buffer.
        device (`torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
    """
    # Recurse if needed
    if value is None:
        return
    # Parameters are renamed by tracing
    model_pipe_module_name = "_".join(param_name.split(".")[:-1])
    model_pipe_tensor_name = param_name.split(".")[-1]
    # Parameters in module
    normal_params = [
        model_pipe_module_name,
        "model_" + model_pipe_module_name,
        "transformer_" + model_pipe_module_name,
    ]
    # Parameters in module._parameters
    moved_params = [
        "moved_" + model_pipe_module_name,
        "moved_model_" + model_pipe_module_name,
        "moved_transformer_" + model_pipe_module_name,
    ]
    tensor_name = None
    for param in normal_params:
        if hasattr(module, param):
            module = getattr(module, param)
            tensor_name = model_pipe_tensor_name
            break
    if tensor_name is None:
        for param in moved_params:
            param = param + "_" + model_pipe_tensor_name
            if param in module._parameters.keys():
                tensor_name = param
                break
    if tensor_name is None:
        return

    if (
        tensor_name not in module._parameters
        and tensor_name not in module._buffers
    ):
        raise ValueError(
            f"{module} does not have a parameter or a buffer named {tensor_name}."
        )
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if (
        old_value.device == torch.device("meta")
        and device not in [None, torch.device("meta")]
        and value is None
    ):
        raise ValueError(
            f"{tensor_name} is on the meta device, we need a `value` to put in on {device}."
        )

    if value is not None:
        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(
            ("torch.uint", "torch.int", "torch.bool")
        ):
            value = value.to(dtype)

    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or device not in [
            None,
            module._parameters[tensor_name].device,
        ]:
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            if param_cls.__name__ == "Int8Params":
                new_value = param_cls(  # type: ignore[misc]
                    new_value, requires_grad=old_value.requires_grad, **kwargs
                ).to(device)
            else:
                new_value = param_cls(  # type: ignore[misc]
                    new_value, requires_grad=old_value.requires_grad
                ).to(device)
            module._parameters[tensor_name] = new_value.to(dtype)  # type: ignore[assignment]
