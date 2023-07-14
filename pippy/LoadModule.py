import gc
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from pippy.utils import _get_binary_filename


TYPICAL_PREFIXES = [
    "model",  # facebook/opt-6.7b
    "transformer",  # bigscience/bloom-7b1
]


def load_checkpoint(
    model: nn.Module,
    index_filename: Union[str, os.PathLike],
    optim: torch.optim.Optimizer = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
    checkpoint_prefix: str = None,
) -> Union[nn.Module, Tuple[nn.Module, torch.optim.Optimizer]]:  # type: ignore
    """
    Load a checkpoint from a model file.
    Args:
        model (`torch.nn.Module`): the model to load the checkpoint into
        index_filename (`Union[str, os.PathLike]`): path to the checkpoint's index (metadata file)
        optim (`torch.optim.Optimizer`): optimizer object to load ckpt state dict into
        device (`torch.device`): the device on which to load the checkpoint
        dtype (`torch.dtype`): the dtype on which to load the checkpoint
        checkpoint_prefix (`str`): the prefix of the checkpoint to load
    Returns:
        The loaded checkpoint model
    Example:
        ```
        checkpoint = load_checkpoint(model, index_filename, device, dtype)
        ```
    """
    checkpoint_folder = os.path.split(index_filename)[0]

    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    if "weight_map" in index:
        index = index["weight_map"]

    prefix_to_test = (
        [checkpoint_prefix] if checkpoint_prefix else TYPICAL_PREFIXES
    )

    file_to_weights = _get_file_to_weight_map(model, index, prefix_to_test)

    used_files = file_to_weights.keys()
    import time

    logging.info(
        f"Timestamp {time.time():.2f} " f"Opening checkpoint: {used_files}"
    )

    for file in used_files:
        file_path = os.path.join(checkpoint_folder, file)
        checkpoint = torch.load(file_path)

        if file in file_to_weights:
            weights = file_to_weights[file]
            for new_name, old_name, clone in weights:
                assert (
                    old_name in checkpoint.keys()
                ), f"{old_name} not in {file}"
                loaded_weight = checkpoint[old_name]
                _set_module_tensor_to_device(
                    model,
                    new_name,
                    device,
                    value=loaded_weight,
                    dtype=dtype,
                    clone=clone,
                )

        del checkpoint
        gc.collect()

    if optim:
        optim.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_folder,
                    _get_binary_filename(dist.get_rank(), is_optim=True),
                )
            )
        )

    if optim:
        return model, optim
    return model


def _get_file_to_weight_map(
    model: nn.Module,
    index: Dict[str, str],
    prefix_to_test: List[str],
) -> Dict[str, List[Tuple]]:
    """
    A helper function to create a mapping from binary checkpoint filename to parameter names
    Args:
        model (`torch.nn.Module`): The model to load weights into
        index (`Dict[str, str]`): The checkpoint index mapping parameter name to binary checkpoint filename
        prefix_to_test (`List[str]`): prefix to try if direct match is not found
    Returns:
        `Dict[str, List[Tuple]]`: A mapping from binary checkpoint filename to list of tuples of parameter names
    Raises:
        RuntimeError: if a parameter name is not found in the checkpoint index
    """
    file_to_weights: Dict[str, List[Tuple]] = {}

    for iterator in [
        model.named_parameters(),
        model.named_buffers(),
    ]:
        for new_name, _ in iterator:
            old_name = model.remap_qualname(new_name)  # type: ignore[operator]
            cp_weight_name, clone_needed = _match_checkpoint_name(
                old_name, index, prefix_to_test
            )
            if cp_weight_name is None:
                raise RuntimeError(
                    f"Weight {new_name} maps to {old_name}, "
                    f"but {old_name} is not found in checkpoint index"
                )
            file = index[cp_weight_name]
            weights = file_to_weights.setdefault(file, [])
            weights.append((new_name, cp_weight_name, clone_needed))

    return file_to_weights


# Some weights like word_embeddings.weight and shared.weight will be used in
# different layers, but these layers may not in the index file, so we can only
# clone the shared weight to their corresponding layers.

TYPICAL_SHARER_WEIGHTS = [
    "lm_head.weight",  # facebook/opt-6.7b
    "encoder_embed_tokens.weight",
]

TYPICAL_SHAREE_WEIGHTS = [
    "decoder.embed_tokens.weight",
    "word_embeddings.weight",
    "shared.weight",
    "wte.weight",
]


def _match_checkpoint_name(
    old_name: str,
    index,
    prefix_to_test: List[str],
) -> Tuple[Optional[str], bool]:
    """
    A helper function to match weight name against those in checkpoint index.
    Args:
        old_name (`str`):
            weight name in original model (retrieved via `remap_qualname()`)
        index (`Dict`?):
            checkpoint index
        prefix_to_test (`List[str]`):
            prefix to try if direct match is not found
    Return:
        weight name in the checkpoint index and whether clone is needed
    Search rule:
        - Exact match, no need to clone
        - Match after prefix, no need to clone
        - Match via shared weight table, clone needed
    """
    if old_name in index.keys():
        return old_name, False

    for prefix in prefix_to_test:
        if (
            old_name.startswith(prefix)
            and old_name[len(prefix) + 1 :] in index.keys()
        ):
            return old_name[len(prefix) + 1 :], False

    if old_name in TYPICAL_SHARER_WEIGHTS:
        for sharee in TYPICAL_SHAREE_WEIGHTS:
            if sharee in index.keys():
                return sharee, True

    return None, False


def _set_module_tensor_to_device(
    module: nn.Module,
    qualname: str,
    device: Optional[torch.device] = None,
    value: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    clone: bool = False,
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
        clone (`bool`, default is False):
            whether to copy the input value.
    """
    # Recurse if needed
    if value is None:
        return

    submod_and_weight = qualname.rsplit(".", 1)
    if len(submod_and_weight) > 1:
        submod = getattr(module, submod_and_weight[0])
        weight = submod_and_weight[1]
    else:
        submod = module
        weight = submod_and_weight[0]

    if weight not in submod._parameters and weight not in submod._buffers:
        raise ValueError(
            f"{submod._get_name()} does not have a parameter or a buffer named {weight}. "
            f"Has instead: {submod._parameters.keys()} and {submod._buffers.keys()}"
        )

    is_buffer = weight in submod._buffers
    old_value = getattr(submod, weight)

    if dtype is None:
        # For compatibility with PyTorch load_state_dict which converts state
        # dict dtype to existing dtype in model
        dtype = old_value.dtype
    elif str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
        # Avoid casting these data types
        dtype = value.dtype

    with torch.no_grad():
        if isinstance(value, torch.Tensor):
            new_value = value.to(device=device, dtype=dtype)
            # In case device and dtype do not change, `new_value` would point to
            # the same storage as `value`, and we would need to explicitly clone
            if clone and new_value.data_ptr() == value.data_ptr():
                new_value = value.clone()
        else:
            # Note: `torch.tensor()` allocates new memory to copy the data of
            # tensor, so clone is taken care of
            new_value = torch.tensor(value, device=device, dtype=dtype)

        if is_buffer:
            submod._buffers[weight] = new_value
        else:
            param_cls = type(submod._parameters[weight])
            kwargs = submod._parameters[weight].__dict__
            if param_cls.__name__ == "Int8Params":
                new_param = param_cls(  # type: ignore[misc]
                    new_value, requires_grad=old_value.requires_grad, **kwargs
                )
            else:
                new_param = param_cls(  # type: ignore[misc]
                    new_value, requires_grad=old_value.requires_grad
                )
            submod._parameters[weight] = new_param
