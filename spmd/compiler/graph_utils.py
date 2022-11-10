import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree

from .log_utils import rank0_info

logger: logging.Logger = logging.getLogger(__name__)

# ------------------------------------------


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


def get_node_tensor_size(node: fx.Node, debug=False) -> int:
    """takes an fx node, and if tensor data available, optionally displays and returns numel"""
    size = None
    tdata = node.meta.get("tensor_meta")
    if tdata is None:
        # assert tdata is not None, f"failed to locate metadata for node {node}"
        return size

    m, n = tdata.shape
    size = m * n

    if debug:
        rank0_info(logger, f"--> tensor of size {size} found for {node=}\n")

    return size


def get_output_node(gm: fx.GraphModule) -> fx.Node:
    """take a graphmodule and returns the graph output node
    we traverse in reverse to expedite it, with the idea that last node should be output"""

    if gm.graph is None:
        raise ValueError(f"missing graph from graph module")
    output_node = None

    for node in reversed(gm.graph.nodes):
        if node.op == OP.OUTPUT:
            output_node = node
            break

    return output_node


def pretty_print_graph(gm: fx.GraphModule, header: str = "graph"):
    """print graphs with surrounding markers to make spotting in console easy"""

    rank0_info(
        logger,
        f"\n\n ======= {header}  ============"
        f"{gm.graph.print_tabular()}\n"
        f"-----------------------------------------------",
    )


def graph_cleanup(gm: fx.GraphModule) -> None:
    """runs the required steps to ensure production-ready graph"""

    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
