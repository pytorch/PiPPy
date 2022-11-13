import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, cast

import graph_utils as gu
import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from functorch.compile import aot_module, make_boxed_func
from graph_utils import OP
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils._pytree import tree_flatten, tree_map

from spmd.tensor import DeviceMesh, DTensor
from spmd.tensor.dispatch import operator_dispatch, propagate_input_sharding
from spmd.tensor.placement_types import Placement, Replicate, Shard, _Partial
from spmd.tensor.redistribute import _redistribute_with_local_tensor


# for prepping for allreduce fusion
@dataclass
class FusionCandidateMap:
    name: str  # TODO - make this a field with auto fill
    node: fx.Node
    dtensor: DTensor
    size: int  # TODO - numel or bytes?
    id: float  # first element of local_tensor
    processed: bool = False


@dataclass
class GraphInfo:
    len: int = 0
    global_buffer: fx.Node = None
    global_buffer_size: int = 0
    output: fx.Node = None
    first: fx.Node = None

    def update_info(self, gm, debug=True):
        """get the len, input and output nodes"""
        graph_len = gm.graph._len
        if not graph_len:
            raise ValueError("Empty graph passed in....")
        self.len = graph_len

        nodelist = gm.graph.nodes

        for i, node in enumerate(nodelist):
            if node.op == "placeholder":
                self.first = node
                break

        self.output = gu.get_output_node(gm)
        assert self.output is not None, f"unable to locate output node"

        if debug:
            print(
                f"updated info - len = {self.len} input = {self.first}, output = {self.output}"
            )


@dataclass
class FusionExtract:
    node_list: list = field(default_factory=lambda: [])
    size: int = 0
    prev_node: fx.Node = None  # node that was before start of section
    next_node: fx.Node = None  # node that was after end of section
    processed: bool = False
    output_name: str = ""
    wait_node: fx.Node = None
    clone_node: fx.Node = None
    all_reduce_node: fx.Node = None

    def __len__(self):
        return len(self.node_list)

    def get_first(self) -> fx.Node:
        """get first node of section"""
        if not len(self):
            return None
        return self.node_list[0]

    def get_last(self) -> fx.Node:
        """get last node of section"""
        if not len(self):
            return None
        return self.node_list[-1]

    def remove_section_nodes(
        self, gm: fx.GraphModule, gi: GraphInfo, debug=True
    ) -> None:

        """remove all nodes in section from the graph"""
        rank = dist.get_rank()

        if not len(self.node_list):
            if debug:
                print(f" --> No internal nodes")
            return None
        if rank == 0:
            print(
                f" =====>>> attempting to clean {len(self.node_list)} items from graph"
            )

        # 1 - remove section (wait comm) from output
        wait_node = self.wait_node
        out_immutable_args = gi.output.args[0]
        output_args = list(out_immutable_args)
        new_args = []
        saved_args = gi.output.args

        if rank == 0:
            print(f" updating output node to remove wait_comm")
            print(f"---> current output node args = {gi.output.args}")
            print(
                f"\n===> output args for remove, type = {type(output_args)}, args = {output_args}\n"
            )
            print(
                f" wait_name = {wait_node.name}, and type first arg = {type(output_args[0])}"
            )

        index = output_args.index(wait_node)
        if rank == 0:
            print(
                f" location of {wait_node} found at {index}, on output node {gi.output}"
            )

        for i, node in enumerate(output_args):
            if i == index:
                # replace with clone tensor node to ensure matching gradient output for now
                new_args.append(self.clone_node.args[0])
                print(
                    f" appending clone node target to output, target = {self.clone_node.args[0]}"
                )
                continue
            new_args.append(node)

        if rank == 0:
            print(f"Created new output list --> {new_args}")

            # output_args.remove(index)
            # assign
        gi.output.args = (new_args,)

        if rank == 0:
            print(
                f"SUCCESS - {gi.output.args}, of type {type(gi.output.args)}, \noriginally {saved_args}, type {type(saved_args)})"
            )
            print(f" type of [0] = {type(gi.output.args[0])}")
            print(f" ***********************************************\n\n")

        # TODO - we continue even if wait_comm not in output node as it means no dependency

        # start here

        # walk from start node to end node

        for i, curr in enumerate(self.node_list):
            if rank == 0:
                print(
                    f"==>>> checking users of node {curr.name}.  users = {curr.users.keys()}"
                )
            # assign empty links
            # curr.args = ()
            # curr.users = {}
        if rank == 0:
            print(f"revisit dependencies....\n")
        for i, curr in enumerate(self.node_list):
            if rank == 0:
                print(
                    f"==>>> RE-checking users of node {curr.name}.  users = {curr.users.keys()}"
                )
            # assign empty links
            # curr.args = ()
            curr.users = {}

        # try to remove now that args (dependencies) are cleaned
        for i, curr in enumerate(reversed(self.node_list)):
            if rank == 0:
                print(
                    f"==>>> checking users of node {curr.name}.  users = {curr.users.keys()}"
                )

            gm.graph.erase_node(curr)

        if rank == 0:

            print(f"++++++++ remove section completed +++++++++++")


def _insert_buffer_node(
    gm: fx.GraphModule, insert_before_node: fx.Node, buffer_size: int
) -> fx.Node:
    """insert a torch.empty node in front of insert_before_node"""
    with gm.graph.inserting_before(insert_before_node):
        new_buffer_node = gm.graph.create_node(
            "call_function",
            target=torch.empty,
            args=(200, 200),  # TODO - need device
        )
    assert new_buffer_node is not None, f"failed to create buffer node"
    return new_buffer_node


# enum for the supported fusion comm types
class Comm_Type(str, Enum):
    allreduce = "allreduce_"
    allgather = "allgather_"
    broadcast = "broadcast_"
    reducescatter = "reduce_scatter_"
    scatter = "scatter_"


@dataclass
class FusionElement:
    comm_type: None
    node_list: list = field(default_factory=lambda: [])
    size: int = 0
    prev_node: fx.Node = None  # node that was before start of section
    next_node: fx.Node = None  # node that was after end of section
    processed: bool = False
    output_name: str = ""
    wait_node: fx.Node = None
    clone_node: fx.Node = None
    all_reduce_node: fx.Node = None


def _scan_graph_for_fusion_elements(
    gm: fx.GraphModule,
    comm_type: Comm_Type = Comm_Type.allreduce,
) -> Optional[list]:
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

    for node in gm.graph.nodes:

        if curr_count < fe_size:
            pattern = fe_pattern[curr_count]
            if rank == 0:
                print(f"{curr_count=}, {node.name=}, {pattern=}\n")
            if node.name.startswith(pattern):
                if rank == 0:
                    print(f"element node found {node.name}")
                curr_node_list.append(node)
                curr_count += 1
                continue
            else:
                curr_count == 0
                curr_node_list.clear()
                continue

        if curr_count == fe_size:
            # should be last node
            if node.name.startswith(fe_pattern[curr_count]):
                if rank == 0:
                    print(f" FOUND Last NODE: {node.name}")
                curr_node_list.append(node)

                fe = FusionElement(
                    comm_type=comm_type, node_list=deepcopy(curr_node_list)
                )

                element_list.append(fe)

            curr_count == 0
            curr_node_list.clear()
            continue

    if rank == 0:
        print(f"======   Comm Nodes Found =====")
        print(f"{element_list=}")

    return element_list


def run_fusion(main_gm: fx.GraphModule) -> bool:
    """main entry into remapping graph for all_reduce fusion"""

    result = False

    rank = dist.get_rank()
    if rank == 0:
        print(f"run fusion entered...")
        # gu.pretty_print_graph(main_gm, "start of fusion pass")

    # get our main graph info
    graph_info = GraphInfo()
    graph_info.update_info(main_gm)

    fe_list = _scan_graph_for_fusion_elements(
        main_gm, comm_type=Comm_Type.allreduce
    )

    # final review print
    gu.graph_cleanup(main_gm)

    if rank == 0:
        gu.pretty_print_graph(main_gm, "final version, fusion pass")

        print(f"***** Graph Info *****")
        print(f"{graph_info}")
        print(f"{main_gm.code}")

    return True

    # isolate all the all_reduce, wait_comm and clone nodes - these should represent the expanded DTensor calls
    comm_nodes = _get_comm_and_clone_nodes(
        main_gm,
    )

    # create master list of all commsections
    comm_sections = []
    offset = 0

    # hardcoded distance between expected nodes for valid allreduce section
    required_delta_clone_all_reduce = 3
    required_delta_all_reduce_wait_comm = 4
    required_total_nodes_in_comm_section = 7

    fusion_sections = []
    fusion_section_sizes = []

    # walk our CommNodes to create FusionExtract list
    for loop_index, clone_node in enumerate(comm_nodes.clone_nodes_map):

        # TODO - add adjustments to curr_index b/c not every clone is a section start
        # for now, we assume every clone node is valid
        curr_index = loop_index  # +offset
        clone_node_index = comm_nodes.clone_nodes_list[curr_index]

        # not all clone nodes are part of a commSection...run some checks
        maybe_allreduce_index = comm_nodes.all_reduce_nodes_list[curr_index]
        maybe_wait_comm_node_index = comm_nodes.wait_nodes_list[curr_index]

        if rank == 0:
            print(
                f"**** loop section indexes: {clone_node_index=}, {maybe_allreduce_index=}, {maybe_wait_comm_node_index=}, \n"
            )

        # is this a comm section?
        actual_delta_all_reduce_wait_comm = (
            maybe_wait_comm_node_index - maybe_allreduce_index
        )
        actual_delta_clone_all_reduce = maybe_allreduce_index - clone_node_index

        if (
            actual_delta_all_reduce_wait_comm
        ) != required_delta_all_reduce_wait_comm:
            if rank == 0:
                print(
                    f"found non matching wait-allreduce distance of {actual_delta_all_reduce_wait_comm}"
                )
            offset += 1
            continue
        if actual_delta_clone_all_reduce != required_delta_clone_all_reduce:
            if rank == 0:
                print(
                    f"found non matching allreduce-clone distance of {actual_delta_clone_all_reduce}"
                )
            offset += 1
            continue

        # we should have a legit Fusion Extract section now

        fusion_extract = FusionExtract()

        all_reduce_node = comm_nodes.index_to_node_map[maybe_allreduce_index]
        wait_comm_node = comm_nodes.index_to_node_map[
            maybe_wait_comm_node_index
        ]

        fusion_extract.prev_node = clone_node.prev
        fusion_extract.next_node = wait_comm_node.next

        fusion_extract.clone_node = clone_node
        fusion_extract.all_reduce_node = all_reduce_node
        fusion_extract.wait_node = wait_comm_node
        fusion_extract.output_name = wait_comm_node.name

        # update size
        curr_size = comm_nodes.comm_size_list[curr_index]
        fusion_extract.size = curr_size
        fusion_section_sizes.append(curr_size)

        # finally add all relevant nodes
        curr = clone_node
        for i in range(required_total_nodes_in_comm_section + 1):
            fusion_extract.node_list.append(curr)
            curr = curr.next

        # safety check
        assert (
            curr.name == fusion_extract.next_node.name
        ), f"last node in section is not as expected {curr.name} vs {fusion_extract.next_node.name}"

        # we have a finished section
        fusion_sections.append(fusion_extract)

    if rank == 0:
        print(f" ((((((( ___________ ))))))))))))))))))")
        for item in fusion_sections:
            print(f"\nfusion_extract {item} = {item}\n")
        print(f"============================")

    """curr = wait_comm_list[-1]
    if rank == 0:
        print(f" ---------- walk subsection -----------------------\n")

        # -------------- working section ---------
        allreduce_sections_list = []

        # -- from clone node, walk forward
        print(f" ------------ clone node, walk forward --------------")
    for outer_index, clone_node in enumerate(clone_list):

        curr_section = FusionExtract()
        curr = clone_node

        curr_section.clone_node = clone_node
        curr_section.prev_node = curr.prev

        if rank == 0:
            print(f" ^^^^^^^^^ section {outer_index} start ^^^^^^^^^^^")

        # we are on the clone node, check the meta size
        # try to check for size:
        size = gu.get_node_tensor_size(curr)
        if size is not None:
            if rank == 0:
                print(f"found size:  {size}")
            curr_section.size = size

        for i in range(9):
            if rank == 0:
                print(
                    f"{outer_index} ---> {curr.name}, {curr.op}, {curr.target}"
                )
            curr_section.node_list.append(curr)

            if curr.name.startswith("wait"):
                if rank == 0:
                    print(f"---->  Found Wait Node {curr}")
                curr_section.wait_node = curr
                curr_section.next_node = curr.next  # the outermostnode
                curr = curr.next
                break
            curr = curr.next

        # save section
        allreduce_sections_list.append(curr_section)

        # should have full list
        if rank == 0:
            print(f"section outer node = {curr.name}")
            print(
                f" ^^^^^^^^^^^^^ section {outer_index} complete ^^^^^^^^^^^^^^\n"
            )

            # check fusion list
            print(f" --------- allreduce section list -------------")
            for item in allreduce_sections_list:
                print(f"{item=}")

            print(
                f"++++++++++= clone walk finish ++++++++++++++++++++++++++++++\n"
            )

    # we have the sections that need to be fused.
    # let's make a buffer
    # in front of first original all reduce node
    curr_section = allreduce_sections_list[0]

    buffer_insert_node = curr_section.get_first()
    """

    if rank == 0:
        print(f"RETURNING EARLY line 439")
    return
    # insert buffer node
    buffer_node = _insert_buffer_node(
        main_gm, buffer_insert_node, buffer_size=200
    )  # TODO - device, adjust size
    graph_info.buffer_node = buffer_node

    # remove first all reduce
    post_section_node = curr_section.get_last().next
    if rank == 0:
        print(f"first non_section node = {post_section_node}")
        print(f" ---> execute remove section...")
    curr_section.remove_section_nodes(
        main_gm,
        graph_info,
    )
    if rank == 0:
        print(f"=========>>>>>> removed self called\n")

        for arg in graph_info.output.args:
            print(f"----> output arg --- {arg}")

    main_gm.recompile()

    # gu.pretty_print_graph(main_gm, "BUFFER ***********************")

    # final review print
    gu.graph_cleanup(main_gm)

    if rank == 0:
        gu.pretty_print_graph(main_gm, "final version, fusion pass")

        print(f"***** Graph Info *****")
        print(f"{graph_info}")
        print(f"{main_gm.code}")

    return True
