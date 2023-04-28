import os
import json
import gc
from typing import Dict, List, Optional, Tuple, Union
import logging

import torch
from torch import nn


def load_checkpoint(
    model: nn.Module,
    index_filename: Union[str, os.PathLike],
    device: torch.device = None,
    dtype: torch.dtype = None,
    prefix: str = "model",
    #shared_weights: Dict[str, str] = None,
):
    checkpoint_folder = os.path.split(index_filename)[0]
    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    if "weight_map" in index:
        index = index["weight_map"]

    file_to_params: Dict[str, List[Tuple]] = {}
    for new_name, param in model.named_parameters():
        old_name = model.remap_qualname(new_name)
        if prefix:
            old_name = old_name[len(prefix)+1 :]
        if old_name not in index.keys():
            logging.warning(
                f"Parameter {new_name} maps to {old_name}, "
                f"but {old_name} is not found in checkpoint index"
            )
            continue
            """
            if old_name in shared_weights.keys():
                old_name = shared_weights[old_name]
            else:
            """
        file = index[old_name]
        param_list = file_to_params.setdefault(file, [])
        param_list.append((new_name, old_name, param))

    file_to_buffers: Dict[str, List[Tuple]] = {}
    for new_name, buffer in model.named_buffers():
        old_name = model.remap_qualname(new_name)
        if prefix:
            old_name = old_name[len(prefix)+1 :]
        if old_name not in index.keys():
            logging.warning(
                f"Buffer {new_name} maps to {old_name}, "
                f"but {old_name} is not found in checkpoint index"
            )
            continue
            """
            if old_name in shared_weights.keys():
                old_name = shared_weights[old_name]
            else:
            """
        file = index[old_name]
        buffer_list = file_to_buffers.setdefault(file, [])
        buffer_list.append((new_name, old_name, buffer))

    set1 = set(file_to_params.keys())
    set2 = set(file_to_buffers.keys())
    used_files = sorted(set1.union(set2))
    logging.info(
        f"Opening checkpoint: {used_files}"
    )

    for file in used_files:
        file_path = os.path.join(checkpoint_folder, file)
        checkpoint = torch.load(file_path)

        for file_to_weights in [file_to_params, file_to_buffers]:
            if file in file_to_weights:
                weights = file_to_weights[file]
                for new_name, old_name, _ in weights:
                    if old_name not in checkpoint.keys():
                        raise ValueError(
                            f"{old_name} not in {file}"
                        )
                    loaded_weight = checkpoint[old_name]
                    set_module_tensor_to_device(
                        model, new_name, device, value=loaded_weight, dtype=dtype
                    )

        handle_shared_weights(model, checkpoint, device=device, dtype=dtype)

        del checkpoint
        gc.collect()

    return model


def set_module_tensor_to_device(
    module: nn.Module,
    qualname: str,
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
        qualname (`str`):
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

    if (
        qualname not in module._parameters
        and qualname not in module._buffers
    ):
        raise ValueError(
            f"{module._get_name()} does not have a parameter or a buffer named {qualname}. "
            f"Has instead: {module._parameters.keys()} and {module._buffers.keys()}"
        )

    is_buffer = qualname in module._buffers
    old_value = getattr(module, qualname)

    if dtype is None:
        # For compatibility with PyTorch load_state_dict which converts state
        # dict dtype to existing dtype in model
        dtype = old_value.dtype
    elif str(value.dtype).startswith(
        ("torch.uint", "torch.int", "torch.bool")
    ):
        # Avoid casting these data types
        dtype = value.dtype

    with torch.no_grad():
        if isinstance(value, torch.Tensor):
            new_value = value.to(device=device, dtype=dtype)
        else:
            new_value = torch.tensor(value, device=device, dtype=dtype)

        if is_buffer:
            module._buffers[qualname] = new_value
        else:
            param_cls = type(module._parameters[qualname])
            kwargs = module._parameters[qualname].__dict__
            if param_cls.__name__ == "Int8Params":
                new_param = param_cls(  # type: ignore[misc]
                    new_value, requires_grad=old_value.requires_grad, **kwargs
                )
            else:
                new_param = param_cls(  # type: ignore[misc]
                    new_value, requires_grad=old_value.requires_grad
                )
            module._parameters[qualname] = new_param


# Some weights like word_embeddings.weight and shared.weight will be used in
# different layers, but these layers may not in the index file, so we can only
# clone the shared weight to their corresponding layers.
def handle_shared_weights(
    module: nn.Module,
    checkpoint,
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    for param_name, param in checkpoint.items():
        if param_name in [
            "word_embeddings.weight",
            "shared.weight",
            "wte.weight",
        ]:
            if hasattr(module, "lm_head"):
                module.lm_head.weight = torch.nn.Parameter(  # type: ignore[union-attr]
                    (param.clone()).to(device).to(dtype)
                )
            if hasattr(module, "encoder_embed_tokens"):
                module.encoder_embed_tokens.weight = torch.nn.Parameter(  # type: ignore[union-attr]
                    (param.clone()).to(device).to(dtype)
                )
            if hasattr(module, "decoder_embed_tokens"):
                module.decoder_embed_tokens.weight = torch.nn.Parameter(  # type: ignore[union-attr]
                    (param.clone()).to(device).to(dtype)
                )
        elif param_name in [
            "decoder.embed_tokens.weight",
        ]:
            # For OPT, the lm_head weight is automatically tied to the embed tokens weight
            if hasattr(module, "lm_head"):
                module.lm_head.weight = torch.nn.Parameter(  # type: ignore[union-attr]
                    (param.clone()).to(device).to(dtype)
                )
