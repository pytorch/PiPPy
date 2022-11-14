from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.fx as fx
from .graph_utils import (
    OP,
    get_node_tensor_numel,
    get_output_node,
    graph_cleanup,
    pretty_print_graph,
)


# enum for the supported fusion comm types
class Comm_Type(str, Enum):
    allreduce = "allreduce_"
    allgather = "allgather_"
    broadcast = "broadcast_"
    reducescatter = "reduce_scatter_"
    scatter = "scatter_"


@dataclass
class FusionElement:
    comm_type: Optional[Comm_Type] = None
    node_list: List = field(default_factory=lambda: [])
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

    def update_info(self, gm: fx.GraphModule, debug: bool = True) -> None:
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

        if debug:
            print(
                f"updated info - len = {self.len} input = {self.first}, output = {self.output}"
            )


def _insert_buffer_node(
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
    comm_type: Comm_Type = Comm_Type.allreduce,
) -> Optional[List[FusionElement]]:
    """scan entire graph for matching sections of CommTensor style expansions
    returns list of FusionElements that match comm_type"""

    element_list = []

    fe_pattern = [
        "clone",
        "_tensor_constant",
        "_tensor_constant",
        comm_type,
        "comm_result",
        "getitem",
        "getitem",
        "wait_comm",
    ]

    fe_size = len(fe_pattern) - 1
    curr_count = 0
    curr_node_list = []
    rank = dist.get_rank()

    for i, node in enumerate(gm.graph.nodes):

        pattern = fe_pattern[curr_count]

        if curr_count < fe_size:

            if node.name.startswith(pattern):
                curr_node_list.append(node)
                curr_count += 1
                continue
            else:
                curr_count == 0
                curr_node_list.clear()
                continue

        if curr_count == fe_size:
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

                # compute size of this fe
                fe.size = get_node_tensor_numel(fe.clone_node)  # type: ignore (confuses optional[node] with node...)
                element_list.append(fe)

            curr_count = 0
            curr_node_list.clear()
            continue

    return element_list


def run_fusion(main_gm: fx.GraphModule) -> bool:
    """main entry into remapping graph for all_reduce fusion"""

    result = False

    # get our main graph info
    graph_info = GraphInfo()
    graph_info.update_info(main_gm)

    # scan graph for all comm sections (fusion elements)
    fe_list = _scan_graph_for_fusion_elements(
        main_gm, comm_type=Comm_Type.allreduce
    )

    # final review print
    graph_cleanup(main_gm)

    pretty_print_graph(main_gm, "final version, fusion pass")

    result = True  # TODO - make this mean something
    return result
