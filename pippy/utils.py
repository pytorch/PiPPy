# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Dict

import torch
import torch.distributed as dist
from torch import fx
from torch.export.unflatten import InterpreterModule


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
            if "device" in node.kwargs and node.kwargs["device"] != new_device:
                logger.debug(
                    f"Changing device of Node {node.name} from {node.kwargs['device']} to {new_device}"
                )
                node.update_kwarg("device", new_device)
                modified = True
        elif node.op == "call_module":
            # Recursively modify "device" in submodules
            submod = gm.get_submodule(node.target)
            if isinstance(submod, torch.fx.GraphModule):
                modify_graph_op_device(submod, new_device)
            elif isinstance(submod, InterpreterModule):
                # If unflattening has been performed, we need to access its graph module by `.graph_module`
                modify_graph_op_device(submod.graph_module, new_device)
            else:
                logger.warning(
                    f"Skipping device modification for submodule {node.target} because it is a {type(submod)}"
                )

    if modified:
        gm.recompile()


class QualnameMapMixin:
    """
    A mixin class to provide qualname remap functionality for both Pipe object
    and submodules
    """

    def __init__(
        self,
        splitter_qualname_map: Dict[str, str] = None,
        tracer_qualname_map: Dict[str, str] = None,
    ):
        self.new_to_old_qualname_mapping: Dict[str, str] = (
            splitter_qualname_map or {}
        )
        self.tracer_qualname_map = tracer_qualname_map

    def remap_qualname(self, qualname: str):
        # TODO: annoying
        if qualname.startswith("split_gm."):
            qualname = qualname[len("split_gm.") :]

        name_before_split = None
        if qualname in self.new_to_old_qualname_mapping:
            name_before_split = self.new_to_old_qualname_mapping[qualname]
        else:
            # The qualname map does not store recursive items, thus,
            # when passed a qualname with leaves, we need to perform longest prefix match
            # Split from the right, one each time
            split_names = qualname.rsplit(".", 1)
            leaf = split_names[-1]
            while len(split_names) > 1:
                prefix = split_names[0]
                if prefix in self.new_to_old_qualname_mapping:
                    old_prefix = self.new_to_old_qualname_mapping[prefix]
                    name_before_split = ".".join([old_prefix, leaf])
                    break
                split_names = prefix.rsplit(".", 1)
                leaf = ".".join([split_names[-1], leaf])

        if name_before_split is None:
            raise RuntimeError(f"Could not find mapping for {qualname}")

        if self.tracer_qualname_map is not None:
            return self.tracer_qualname_map[name_before_split]
        else:
            return name_before_split
