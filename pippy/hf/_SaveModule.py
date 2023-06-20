import torch.distributed as dist
import pippy

from itertools import chain
import tempfile
import logging
import shutil
import json
import os

CKPT_INDEX_JSON_FILENAME = "pytorch_model.bin.index.json"

logger = logging.getLogger(__name__)


def _copy_with_metadata(source: str, target: str) -> None:
    """
    Copy file with all its permisions and metadata.

    Args:
        source (str): path to source file
        target (str): path to copy to
    """
    shutil.copy2(source, target)
    st = os.stat(source)
    os.chown(target, st.st_uid, st.st_gid)


def _atomic_write(file_contents: str, target_file_path: str, mode="w") -> None:
    """
    Atomically writes `file_contents` into `target_file_path`.

    Args:
        file_contents (str): contents to write to file
        target_file_path (str): path to write to
        mode (str, optional): mode to write file with. Defaults to "w". Only "w" and "a" are supported.
    """
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, dir=os.path.dirname(target_file_path)
    )
    try:
        if os.path.exists(target_file_path):
            _copy_with_metadata(target_file_path, temp_file.name)
        with open(temp_file.name, mode) as f:
            f.write(file_contents)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_file.name, target_file_path)
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass


def _save_index(
    pipe: pippy.fx.GraphModule,
    ckpt_index_filename: str = CKPT_INDEX_JSON_FILENAME,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Saves index file describing location of weights in checkpoint.

    Args:
        pipe (pippy.fx.GraphModule): pipeline graph module with weights to save
        ckpt_index_filename (str, optional): name of index file. Defaults to "pytorch_model.bin.index.json".
        checkpoint_dir (str, optional): directory to save checkpoint to. Defaults to "checkpoints".
    """
    index_dict = {}
    total_size = 0
    index_dict["metadata"] = {"total_size": total_size}

    weight_map = {}
    for idx, (submod_name, submod) in enumerate(pipe.split_gm.named_children()):
        params_buffers = chain(
            submod.named_parameters(), submod.named_buffers()
        )
        for param_name, _ in params_buffers:
            old_name = submod.remap_qualname(param_name)

            binary_filename = _get_binary_filename(idx)
            weight_map[old_name] = binary_filename

    index_dict["weight_map"] = weight_map

    json_str = json.dumps(index_dict, indent=4)

    filepath = os.path.join(checkpoint_dir, ckpt_index_filename)

    # create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    _atomic_write(json_str, filepath)

    logger.info(f"Saved index file to {filepath}")


def _get_binary_filename(cur_idx: int) -> str:
    """
    Gets filename for pytorch checkpoint binary based on current index and world size.

    Args:
        cur_idx (int): current device index

    Returns:
        str: checkpoint filename
    """
    cur_idx = str(cur_idx + 1).zfill(5)
    world_size = str(dist.get_world_size()).zfill(5)

    return f"pytorch_model-{cur_idx}-of-{world_size}.bin"
