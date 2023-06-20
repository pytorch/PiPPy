import torch.distributed as dist
import pippy

import json
import os

CKPT_INDEX_JSON_FILENAME = "pytorch_model.bin.index.json"


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
        for param_name, _ in submod.named_parameters():
            old_name = submod.remap_qualname(param_name)

            binary_filename = _create_binary_filename(idx)
            weight_map[old_name] = binary_filename
    index_dict["weight_map"] = weight_map

    json_str = json.dumps(index_dict, indent=4)

    filepath = os.path.join(checkpoint_dir, ckpt_index_filename)

    # create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    with open(filepath, "w") as f:
        f.write(json_str)


def _create_binary_filename(cur_idx: int) -> str:
    """
    Creates filename for pytorch checkpoint binary based on current index and world size.

    Args:
        cur_idx (int): current device index

    Returns:
        str: checkpoint filename
    """
    cur_idx = str(cur_idx + 1).zfill(5)
    world_size = str(dist.get_world_size()).zfill(5)

    return f"pytorch_model-{cur_idx}-of-{world_size}.bin"
