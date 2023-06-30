import json
import logging
import os
import tempfile
from itertools import chain

from typing import Dict

import torch
import torch.distributed as dist

from pippy.IR import Pipe

CKPT_INDEX_JSON_FILENAME = "pytorch_model.bin.index.json"

DTYPE_SIZES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


def _get_param_size(param: torch.Tensor) -> int:
    """
    Returns a tensor's size in bytes

    Args:
        param(`torch.Tensor`): torch tensor
    """
    return param.numel() * DTYPE_SIZES[param.dtype]


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
    index_dict = {}
    total_size = 0

    weight_map: Dict[str, str] = {}
    for idx, (_, submod) in enumerate(pipe.split_gm.named_children()):  # type: ignore
        # chain both params and buffers generators together
        params_buffers = chain(
            submod.named_parameters(), submod.named_buffers()
        )
        for param_name, param in params_buffers:
            old_name = submod.remap_qualname(param_name)  # type: ignore

            binary_filename = _get_binary_filename(idx)

            #  add ckpt size once
            if old_name not in weight_map:
                total_size += _get_param_size(param)  # type: ignore

            weight_map[old_name] = binary_filename

    index_dict["metadata"] = {"total_size": total_size}  # type: ignore
    index_dict["weight_map"] = weight_map  # type: ignore

    # serialize json
    json_str = json.dumps(index_dict, indent=4)

    filepath = os.path.join(checkpoint_dir, ckpt_index_filename)

    # create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # write index file atomically to avoid partial/corrupted writes
    _atomic_write(json_str, filepath)

    logging.info(f"Saved index file to {filepath}")


def _get_binary_filename(cur_idx: int, is_optim: bool = False) -> str:  # type: ignore[valid-type]
    """
    Gets filename for pytorch checkpoint binary based on current index and world size.

    Args:
        cur_idx (int): current device index
        is_optim (bool): True if generating binary filename for optimizer,
                         False otherwise

    Returns:
        str: checkpoint filename
    """
    idx = str(cur_idx + 1).zfill(5)
    world_size = str(dist.get_world_size()).zfill(5)

    state_type = "optim" if is_optim else "model"

    return f"pytorch_{state_type}-{idx}-of-{world_size}.bin"


def _save_params(submod: torch.nn.Module, checkpoint_dir: str) -> None:
    """
    writes `module`'s parameters and buffers to disk.

    Args:
        submod(`Pipe`): a submodule of the model's graph
        checkpoint_dir(`str`): where to keep the checkpoint binaries
    """
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    filepath = os.path.join(
        checkpoint_dir, _get_binary_filename(dist.get_rank())
    )
    torch.save(
        {
            submod.remap_qualname(param_name): param  # type: ignore
            for param_name, param in submod.state_dict().items()
        },
        filepath,
    )


def _save_optim_state(
    optimizer: torch.optim.Optimizer, checkpoint_dir: str
) -> None:
    """
    saves `optimizer`'s state_dict to disk.

    Args:
        optimizer(`torch.optim.Optimizer`): pytorch optimizer
        checkpoint_dir(`str`): where to keep the checkpoint binaries
    """
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    filepath = os.path.join(
        checkpoint_dir, _get_binary_filename(dist.get_rank(), is_optim=True)
    )
    # save optimizer state directly
    torch.save(optimizer.state_dict(), filepath)


def save_checkpoint(
    stage: Pipe,
    checkpoint_dir: str = "checkpoints",
    optimizer: torch.optim.Optimizer = None,
) -> None:
    """
    Save the entire model's(`stage`) metadata in an index file and the `submod`
    parameters in `checkpoint_dir`

    Args:
        stage(`Pipe`): model pipeline graph
        checkpoint_dir(`str`): directory where to save the index file and params binaries
                              defaults to `checkpoints`
        optimizer(`torch.optim.Optimizer`): optimizer whose state dict is to be saved
    """
    # write index file in rank 0
    if dist.get_rank() == 0:
        _save_index(stage, checkpoint_dir=checkpoint_dir)

    _save_params(stage.submod, checkpoint_dir)  # type: ignore
    # save optimizer state, if passed
    if optimizer:
        _save_optim_state(optimizer, checkpoint_dir)  # type: ignore
