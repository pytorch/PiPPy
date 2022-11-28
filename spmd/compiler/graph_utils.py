import logging
from enum import Enum
from typing import Dict, Optional, Tuple, Any

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


def get_node_tensor_numel_shape(
    node: fx.Node,
) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """takes an fx node, and if tensor data available, optionally displays and returns numel"""
    size = None
    shape = None
    tdata = node.meta.get("tensor_meta")
    if tdata is None:
        # assert tdata is not None, f"failed to locate metadata for node {node}"
        return size, shape

    if len(tdata.shape) == 1:
        m = tdata.shape
        shape = (m,)
    else:
        m, n = tdata.shape
        size = m * n
        shape = (m, n)

    rank0_debug(
        logger,
        f"--> tensor of size {size} and shape {shape} found for {node=}\n",
    )

    return size, shape


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


def get_all_nodes_of_type(
    gm: fx.GraphModule,
    node_type: OP,
    starts_with: Optional[str] = None,
    require_meta: bool = False,
) -> Dict[str, fx.Node]:

    results_dict = {}

    for node in gm.graph.nodes:

        starts_with_met = False
        require_meta_met = False

        if node.op != node_type:
            continue

        if starts_with is not None:
            if node.name.startswith(starts_with):
                starts_with_met = True
        elif starts_with is None:
            starts_with_met = True

        if require_meta:
            metadata = node.meta.get("tensor_meta")
            if metadata:
                require_meta_met = True
        elif not require_meta:
            require_meta_met

        # add qualifying node
        if starts_with_met and require_meta_met:
            results_dict[node.name] = node

    return results_dict


def graph_cleanup(gm: fx.GraphModule, remove_dead_code: bool = True) -> None:
    """runs the required steps to ensure production-ready graph.
    note - per the fx docs, eliminate dead code is not very precise.
    Hence, the flag to make this step optional."""

    gm.graph.lint()
    if remove_dead_code:
        gm.graph.eliminate_dead_code()
    gm.recompile()
