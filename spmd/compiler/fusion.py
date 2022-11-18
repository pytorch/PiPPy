import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Union

import torch
import torch.fx as fx

from .graph_utils import (
    OP,
    get_node_tensor_numel,
    get_output_node,
    graph_cleanup,
    pretty_print_graph,
)
from .log_utils import rank0_debug, rank0_info
from functools import partial

logger: logging.Logger = logging.getLogger(__name__)

_debug = partial(rank0_debug, logger)
_info = partial(rank0_info, logger)

# enum for the supported fusion comm types
class CommType(str, Enum):
    allreduce = "allreduce_"
    allgather = "allgather_"
    broadcast = "broadcast_"
    reducescatter = "reduce_scatter_"
    scatter = "scatter_"


@dataclass
class FusionElement:
    """This class tracks the nodes for a DTensor expanded communication collective in the graph"""

    in_graph: bool = False
    comm_type: Optional[CommType] = None
    node_list: Optional[List[fx.Node]] = field(default_factory=lambda: [])  # type: ignore
    size: int = 0
    prev_node: Optional[fx.Node] = None  # node that was before start of section
    next_node: Optional[fx.Node] = None  # node that was after end of section
    processed: bool = False
    output_name: str = ""
    clone_node: Optional[fx.Node] = None
    comm_node: Optional[fx.Node] = None
    wait_node: Optional[fx.Node] = None


@dataclass
class GraphInfo:
    len: int = 0
    global_buffer: Optional[fx.Node] = None
    global_buffer_size: int = 0
    output: Optional[fx.Node] = None
    first: Optional[fx.Node] = None

    def update_info(self, gm: fx.GraphModule) -> None:
        """get the len, input and output nodes"""
        graph_len = gm.graph._len
        if not graph_len:
            raise ValueError("Empty graph passed in....")
        self.len = graph_len

        nodelist = gm.graph.nodes

        for i, node in enumerate(nodelist):
            if node.op == OP.PLACEHOLDER:
                self.first = node
                break

        self.output = get_output_node(gm)
        assert (
            self.output is not None
        ), f"unable to locate output node in gm {gm.graph}"

        rank0_debug(
            logger,
            f"updated graph_info - len = {self.len} input = {self.first}, output = {self.output}",
        )


def _insert_fusion_buffer_node(
    gm: fx.GraphModule, insert_before_node: fx.Node, buffer_size: Iterable[int]
) -> fx.Node:
    """insert a torch.empty node in front of insert_before_node"""
    with gm.graph.inserting_before(insert_before_node):
        new_buffer_node = gm.graph.create_node(
            OP.CALL_FUNCTION,
            target=torch.empty,
            # TODO - need device from DTensor to put buffer on gpu
            args=tuple(buffer_size),
        )
    assert (
        new_buffer_node is not None
    ), f"failed to create buffer node, size={buffer_size}"

    return new_buffer_node


def _scan_graph_for_fusion_elements(
    gm: fx.GraphModule,
    comm_type: CommType = CommType.allreduce,
) -> Optional[List[FusionElement]]:
    """scan entire graph for matching sections of CommTensor style expansions
    returns list of FusionElements that match CommType"""

    element_list = []

    fe_sequence = [
        "clone",
        "_tensor_constant",
        "_tensor_constant",
        comm_type,
        "comm_result",
        "getitem",
        "getitem",
        "wait_comm",
    ]

    fe_size = len(fe_sequence) - 1
    index = 0
    curr_node_list = []

    for i, node in enumerate(gm.graph.nodes):
        pattern = fe_sequence[index]

        if index < fe_size:
            if node.name.startswith(pattern):
                curr_node_list.append(node)
                index += 1
                continue
            else:
                index = 0
                curr_node_list.clear()
                continue

        elif index == fe_size:
            # should be last node
            if node.name.startswith(pattern):
                curr_node_list.append(node)

                fe = FusionElement(
                    comm_type=comm_type, node_list=deepcopy(curr_node_list)
                )

                # need to fully populate this fe...
                # we will be removing/rewriting the node list so we save prev and next
                fe.prev_node = curr_node_list[0].prev
                fe.next_node = node.next

                fe.output_name = node.name
                fe.wait_node = node

                fe.clone_node = curr_node_list[0]
                fe.comm_node = curr_node_list[3]

                fe.size = get_node_tensor_numel(fe.clone_node)  # type: ignore
                element_list.append(fe)

            index = 0
            curr_node_list.clear()
            continue

    return element_list


def _remove_gradient_tensor_clones(
    gm: fx.GraphModule, comm_type=CommType.allreduce
) -> int:
    """
    Optimizes away any duplicate gradient tensor nodes from DTensor
    comm insertion in the provided graph.

    Returns - total count of clone tensor nodes removed
    """

    count_clones_removed = 0
    sequence: list[Union[str, CommType]] = [
        "clone",
        "_tensor_constant0",
        "_tensor_constant1",
        comm_type,
    ]

    len_sequence: int = len(sequence) - 1
    index: int = 0
    clone_node: fx.Node = None  # type: ignore
    comm_node: fx.Node = None  # type: ignore

    for node in gm.graph.nodes:

        if node.op == OP.PLACEHOLDER:
            index = 0
            continue

        pattern = sequence[index]

        if (
            index == 0
            and node.op == OP.CALL_FUNCTION
            and node.name.startswith(pattern)
        ):
            clone_node = node
            index += 1
            continue

        elif index < len_sequence and node.name.startswith(pattern):
            index += 1
            continue

        elif (
            index == len_sequence
            and node.op == OP.CALL_FUNCTION
            and node.name.startswith(pattern)
        ):
            # found matching clone/comm sequence...
            grad_tensor_node = clone_node.args[0]
            comm_node = node

            comm_node.update_arg(0, [grad_tensor_node])

            # reset for next series
            count_clones_removed += 1
            index = 0
        else:
            # failed sequence
            index = 0

    if count_clones_removed:
        graph_cleanup(gm)

    return count_clones_removed


def run_comm_fusion(gm: fx.GraphModule) -> fx.GraphModule:
    """main entry into graph optimizations for all_reduce fusion"""

    # optimize out any clone gradient nodes before we start
    removed_tensor_clones = _remove_gradient_tensor_clones(gm)

    _info(
        f" removed {removed_tensor_clones} cloned gradient tensors from graph"
    )

    return gm
