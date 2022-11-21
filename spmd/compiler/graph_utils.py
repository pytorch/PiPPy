import logging
from enum import Enum
from typing import Dict, Optional

import torch.fx as fx

from .log_utils import rank0_debug, rank0_info

logger: logging.Logger = logging.getLogger(__name__)

# ------------------------------------------


class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


def create_graph_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    """utility to put graph module into a node map for easier adjustments"""
    mapping = {}
    for node in gm.graph.nodes:
        mapping[node.name] = node
    return mapping


def get_node_tensor_numel(node: fx.Node) -> Optional[int]:
    """takes an fx node, and if tensor data available, optionally displays and returns numel"""
    size = None
    tdata = node.meta.get("tensor_meta")
    if tdata is None:
        # assert tdata is not None, f"failed to locate metadata for node {node}"
        return size

    m, n = tdata.shape
    size = m * n

    rank0_debug(logger, f"--> tensor of size {size} found for {node=}\n")

    return size


def get_output_node(gm: fx.GraphModule) -> Optional[fx.Node]:
    """take a graphmodule and returns the graph output node
    we traverse in reverse to expedite it, with the idea that last node should be output"""

    if gm.graph is None:
        raise ValueError("missing graph from graph module")
    output_node = None

    for node in reversed(gm.graph.nodes):
        if node.op == OP.OUTPUT:
            output_node = node
            break

    return output_node


def pretty_print_graph(gm: fx.GraphModule, header: str = "graph") -> None:
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
