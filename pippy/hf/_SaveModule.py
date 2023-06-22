import torch.distributed as dist
from pippy.IR import Pipe
import torch

from itertools import chain
import tempfile
import logging

import json
import os

CKPT_INDEX_JSON_FILENAME = "pytorch_model.bin.index.json"

DTYPE_SIZES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


def _atomic_write(file_contents: str, target_file_path: str, mode="w") -> None:
    """
    Atomically writes `file_contents` into `target_file_path`.
    Args:
        file_contents (str): contents to write to file
        target_file_path (str): path to write to
        mode (str, optional): mode to write file with. Defaults to "w". Only "w" and "a" are supported.
    """
    # create tempfile as `move` ops aren't guaranteed to be atomic when between different file systems
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=os.path.dirname(target_file_path),
    )
    try:
        with open(temp_file.name, mode) as f:
            f.write(file_contents)
            # sync in-memory state with storage device
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file.name, target_file_path)
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                raise RuntimeError(f"Failed to delete {temp_file.name}")


def _get_module_size(pipe: Pipe) -> int:
    """
    Returns the size of a module in bytes

    Args:
        pipe(`Pipe`): pipeline graph module

    Returns:
        int: size of the module in bytes
    """
    return sum(
        param.numel() * DTYPE_SIZES[param.dtype]
        for param in pipe.parameters()
        if param.requires_grad is True
    )


def _save_index(
    pipe: Pipe,
    ckpt_index_filename: str = CKPT_INDEX_JSON_FILENAME,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Saves index file describing location of weights in checkpoint.

    Args:
        pipe (Pipe): pipeline graph module with weights to save
        ckpt_index_filename (str, optional): name of index file. Defaults to "pytorch_model.bin.index.json".
        checkpoint_dir (str, optional): directory to save checkpoint to. Defaults to "checkpoints".
    """
    index_dict = {
        "metadata": {
            "total_size": 0,
        },
        "weight_map": {},
    }

    weight_map = {}
    for idx, (_, submod) in enumerate(pipe.split_gm.named_children()):  # type: ignore
        index_dict["metadata"]["total_size"] += _get_module_size(submod)

        # chain both params and buffers generators together
        params_buffers = chain(
            submod.named_parameters(), submod.named_buffers()
        )
        for param_name, _ in params_buffers:
            old_name = submod.remap_qualname(param_name)  # type: ignore

            binary_filename = _get_binary_filename(idx)
            weight_map[old_name] = binary_filename

    index_dict["weight_map"] = weight_map

    # serialize json
    json_str = json.dumps(index_dict, indent=4)

    filepath = os.path.join(checkpoint_dir, ckpt_index_filename)

    # create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # write index file atomically to avoid partial/corrupted writes
    _atomic_write(json_str, filepath)

    logging.info(f"Saved index file to {filepath}")


def _get_binary_filename(cur_idx: int) -> str:  # type: ignore[valid-type]
    """
    Gets filename for pytorch checkpoint binary based on current index and world size.

    Args:
        cur_idx (int): current device index

    Returns:
        str: checkpoint filename
    """
    idx = str(cur_idx + 1).zfill(5)
    world_size = str(dist.get_world_size()).zfill(5)

    return f"pytorch_model-{idx}-of-{world_size}.bin"
