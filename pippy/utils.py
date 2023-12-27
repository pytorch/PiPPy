# Copyright (c) Meta Platforms, Inc. and affiliates
import logging

import torch
import torch.distributed as dist
from torch import fx


logger = logging.getLogger(__name__)


def flatten_args_detach(args):
    flat_detached_args = []

    def extract_tensor_args(a):
        nonlocal flat_detached_args
        if isinstance(a, torch.Tensor):
            val = a.detach().requires_grad_(a.requires_grad)
            flat_detached_args.append(val)
            return val
        else:
            flat_detached_args.append(a)
            return a

    """
    def dont_traverse_size(a):
        return type(a) != torch.Size
    """

    new_args = fx.node.map_aggregate(
        args,
        extract_tensor_args,  # dont_traverse_size
    )

    return new_args, flat_detached_args


def flatten_args(args):
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        flat_args.append(a)
        return a

    """
    def dont_traverse_size(a):
        return type(a) != torch.Size
    """

    fx.node.map_aggregate(
        args,
        extract_tensor_args,  # dont_traverse_size
    )

    return flat_args


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


def modify_graph_op_device(
    gm: torch.fx.GraphModule,
    new_device: torch.device,
):
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if "device" in node.kwargs:
                node.update_kwarg("device", new_device)
                logger.debug(
                    f"Changed device of Node {node.name} to {new_device}"
                )
                modified = True

    if modified:
        gm.recompile()
