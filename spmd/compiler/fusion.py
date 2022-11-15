import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional

import torch
import torch.fx as fx

from .graph_utils import (
    OP,
    get_node_tensor_numel,
    get_output_node,
    graph_cleanup,
    pretty_print_graph,
)
from .log_utils import rank0_debug

logger: logging.Logger = logging.getLogger(__name__)

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
            args=tuple(buffer_size),  # TODO - need device
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
        CommType,
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
                fe.prev_node = fe.node_list[0].prev
                fe.next_node = node.next

                fe.output_name = node.name

                fe.clone_node = curr_node_list[0]  # fe.node_list[0]
                fe.comm_node = fe.node_list[3]
                fe.wait_node = node

                fe.size = get_node_tensor_numel(fe.clone_node)  # type: ignore
                element_list.append(fe)

            index = 0
            curr_node_list.clear()
            continue

    return element_list


def run_comm_fusion(gm: fx.GraphModule) -> bool:
    """main entry into remapping graph for all_reduce fusion"""

    result = False

    # get our main graph info
    graph_info = GraphInfo()
    graph_info.update_info(gm)

    # scan graph for all comm sections (fusion elements)
    fe_list = _scan_graph_for_fusion_elements(gm, comm_type=CommType.allreduce)

    # final review print
    graph_cleanup(gm)

    pretty_print_graph(gm, "final version, fusion pass")

    result = True  # TODO - make this mean something
    return result
