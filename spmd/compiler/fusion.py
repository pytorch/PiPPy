import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import TensorMetadata

from .graph_utils import (
    OP,
    create_name_to_node_map,
    get_node_tensor_numel_shape,
    get_output_node,
    graph_cleanup,
)
from .log_utils import rank0_debug

logger: logging.Logger = logging.getLogger(__name__)
_debug = partial(rank0_debug, logger)  # type: ignore

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

    # monitor if this FusionElement is in the main graph or removed as part of fusion
    in_graph: bool = False
    # has gone through the fusion policy process
    processed: bool = False
    size: Optional[int] = 0
    shape: Optional[Iterable[[int], [int]]] = field(default_factory=lambda: [])  # type: ignore
    comm_type: Optional[CommType] = None
    node_list: Optional[List[fx.Node]] = field(default_factory=lambda: [])  # type: ignore
    prev_node: Optional[fx.Node] = None  # node that was before start of section
    output_name: str = ""
    comm_node: Optional[fx.Node] = None  # type: ignore
    wait_node: Optional[fx.Node] = None  # type: ignore
    grad_tensor_node: Optional[fx.Node] = None  # type: ignore

    def _get_next_node(
        self,
    ) -> fx.Node:
        """Get the next node after this FE section"""
        next_node = self.node_list[-1].next  # type: ignore

        assert (
            next_node is not None
        ), f"failed to get valid next node after {self.node_list[-1].name}"  # type: ignore
        return next_node


@dataclass
class GraphInfo:
    """Provides a home for global aspects of this graph.
    Currently tracks first and last node, len of the graph and
    the location and size of the global buffer node
    """

    # starting len of the graph
    len: int = 0
    # total count of initial fusion elements
    num_starting_fe: int = 0
    # list of all FusionElements in the graph
    fe_list: Optional[Iterable[FusionElement]] = None  # type: ignore
    # max memory needed for fusion buffer
    peak_memory_required: int = 0
    # node housing global buffer for fusion comms
    global_buffer_node: Optional[fx.Node] = None  # type: ignore
    # size of the global buffer
    global_buffer_size: int = 0
    # real buffer (not node) used for tracing fusion subgraphs
    tracing_buffer: Optional[torch.Tensor] = None  # type: ignore
    # first node in graph (head)
    first: Optional[fx.Node] = None
    # last node in graph (tail / output)
    output: Optional[fx.Node] = None
    # offset to comm node within a FusionElement sequence
    fe_offset_to_comm_node: Optional[int] = None

    def update_info(self, gm: fx.GraphModule) -> None:
        """Get the len, input and output nodes"""
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
        ), f"Unable to locate output node in gm {gm.graph}"

        rank0_debug(
            logger,
            f"Updated graph_info - len = {self.len} input = {self.first}, output = {self.output}",
        )


def _insert_fusion_buffer_node(
    gm: fx.GraphModule,
    buffer_size: int,
    gi: Optional[GraphInfo] = None,
) -> fx.Node:
    """Insert a torch.empty node for the global buffer.
    defaults to first node after placeholder nodes.
    appends to GlobalInfo if passed in"""

    # default to inserting just after last placeholder node
    for node in gm.graph.nodes:
        if node.op == OP.PLACEHOLDER:
            continue
        insert_before_node = node
        break

    # TODO - fix with correct rank - needs to match with higher DTensor device

    rank = dist.get_rank()
    if torch.distributed.is_initialized():
        torch.cuda.set_device(rank)
    rank_device = torch.cuda.current_device()

    with gm.graph.inserting_before(insert_before_node):
        new_buffer_node = gm.graph.create_node(
            OP.CALL_FUNCTION,
            target=torch.empty,
            # TODO - need device from DTensor to put buffer on gpu
            args=(buffer_size,),
            kwargs={"device": rank_device},
        )
    assert (
        new_buffer_node is not None
    ), f"failed to create buffer node, size={buffer_size}"

    if gi is not None:
        gi.global_buffer_node = new_buffer_node
        gi.global_buffer_size = buffer_size

    return new_buffer_node


def _scan_graph_for_fusion_elements(
    gi: GraphInfo,
    gm: fx.GraphModule,
    comm_type: CommType = CommType.allreduce,
) -> Optional[List[FusionElement]]:
    """Scan entire graph for matching sections of CommTensor style expansions
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

    # depth to reach comm_node
    if gi.fe_offset_to_comm_node is None:
        for depth, item in enumerate(fe_sequence):
            if isinstance(item, CommType):
                gi.fe_offset_to_comm_node = depth
                break
    assert (
        gi.fe_offset_to_comm_node is not None
    ), f"Unable to locate comm node in {fe_sequence}."

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
                    comm_type=comm_type, node_list=curr_node_list[:]
                )

                # need to fully populate this fe...
                # we will be removing/rewriting the node list so we save prev and next
                fe.prev_node = curr_node_list[0].prev

                fe.output_name = node.name
                fe.wait_node = node
                fe.comm_node = curr_node_list[gi.fe_offset_to_comm_node]  # type: ignore

                fe.grad_tensor_node = fe.comm_node.args[0][0]  # type: ignore

                size, shape = get_node_tensor_numel_shape(fe.grad_tensor_node)  # type: ignore
                fe.size = size
                fe.shape = shape

                element_list.append(fe)

            index = 0
            curr_node_list.clear()
            continue

    return element_list


def _copy_fe_to_buffer(
    gi: GraphInfo, gm: fx.GraphModule, copy_list: list[FusionElement]
) -> None:
    """First half of fusion - move desired items to buffer and create graph"""
    buffer_node = gi.global_buffer_node
    buffer_size = gi.global_buffer_size

    num_fusion_elements = len(copy_list)

    def copy_to_buffer(
        buffer: torch.Tensor, tensor_list: List[torch.Tensor]
    ) -> torch.Tensor:
        offset = 0
        for t in tensor_list:
            size = t.numel()
            buffer[offset : offset + size] = t.view(-1)
            offset += size
        return buffer

    # setup dummy vars
    buffer = None
    if gi.tracing_buffer is None:
        buffer = torch.empty(buffer_size)
        gi.tracing_buffer = buffer
    elif gi.tracing_buffer:
        buffer = gi.tracing_buffer

    tlist = []
    for item in copy_list:
        a = torch.zeros_like(item)  # type: ignore
        tlist.append(a)

    load_gm = make_fx(copy_to_buffer)(buffer, tlist)

    subnodemap = create_name_to_node_map(load_gm)

    # update load loop to use main graph items
    fn_list = []
    pl_list = []
    for node in load_gm.graph.nodes:
        if node.op == OP.PLACEHOLDER:
            pl_list.append(node)
        elif node.op == OP.CALL_FUNCTION:
            fn_list.append(node)

    # create placeholder remapping
    pl_map: Dict[fx.Node, fx.Node] = {}
    pl_map[pl_list[0]] = gi.global_buffer_node  # type: ignore

    for i in range(num_fusion_elements):
        # pl map remaps traced placeholders used in copy graph to main graph grad tensors
        pl_map[pl_list[i + 1]] = copy_list[i].grad_tensor_node  # type: ignore

    insert_node = copy_list[-1].comm_node
    value_remap: Dict[fx.Node, fx.Node] = {}

    def remap_copy_args(in_node: fx.Node) -> fx.Node:
        out_node = in_node
        if in_node in pl_map:
            out_node = pl_map[in_node]  # type: ignore
        elif in_node in value_remap:
            out_node = value_remap[in_node]
        return out_node

    with gm.graph.inserting_before(insert_node):
        for innernode in load_gm.graph.nodes:
            if innernode.op in [OP.PLACEHOLDER, OP.OUTPUT]:
                continue
            value_remap[innernode] = gm.graph.node_copy(
                innernode, remap_copy_args
            )

    _update_new_copy_nodes_users(value_remap)

    # update allreduce to use buffer
    # (we currently don't) have to make our own all_reduce/comm_wait section
    # # TODO - pg group matching
    # _build_buffer_comm_graph(gm, gi)

    buffer_comm_node = copy_list[-1].comm_node
    buffer_comm_node.update_arg(0, [buffer_node])  # type: ignore


def _build_buffer_comm_graph(
    gi: GraphInfo, gm: fx.GraphModule
) -> fx.GraphModule:
    """This function is only a stub atm, for cases where we have
    to make our own all_reduce and wait subgraph for buffer. Wrapping with
    CommTensor is required to complete.
    """
    # from torch.distributed._spmd.comm_tensor import CommTensor
    from torch.distributed.distributed_c10d import ReduceOp, _get_default_group

    buffer_size = gi.global_buffer_size

    def dummy_add(
        grad_buffer: torch.Tensor, zero: torch.Tensor
    ) -> torch.Tensor:
        return grad_buffer + zero

    grad_buffer: torch.Tensor = torch.empty(buffer_size)
    zero: torch.Tensor = torch.zeros_like(grad_buffer)

    traced_add = make_fx(dummy_add)(grad_buffer, zero)

    # TODO - needs to match to DTensor PG
    pg = _get_default_group()
    tensor: torch.Tensor
    op: ReduceOp = ReduceOp.SUM  # type: ignore[assignment]
    async_op: bool = False

    # work_handle = all_reduce(grad_buffer, op=op, group=pg, async_op=async_op)
    return traced_add


def _scatter_results_from_buffer(
    gi: GraphInfo, gm: fx.GraphModule, fe_list: List[FusionElement]
) -> None:
    """After comm event with buffer, scatter results back to original fe grad tensors"""

    buffer_node = gi.global_buffer_node
    buffer_size = gi.global_buffer_size

    scatter_list = fe_list
    num_fe_items = len(scatter_list)

    def scatter_from_buffer(
        buffer: torch.Tensor, scatter_list: List[torch.Tensor]
    ) -> torch.Tensor:
        offset = 0
        for t in scatter_list:
            numel = t.numel()
            shaper = buffer[offset : offset + numel].view(t.shape)
            t.copy_(shaper)  # buffer[offset : offset + numel].view(t.shape() ))
            offset += numel
        return buffer

    buffer = gi.tracing_buffer
    assert buffer is not None, f" missing global tracing buffer in {gi}"
    buffer_shape = buffer.shape

    tlist = []
    for item in scatter_list:
        shape = item.shape

        a = torch.zeros(item.shape[0], item.shape[1])  # type: ignore

        tlist.append(a)  # clone().detach())

    scatter_sg = make_fx(scatter_from_buffer)(buffer, tlist)

    pl_list = []

    for node in scatter_sg.graph.nodes:
        if node.op == OP.PLACEHOLDER:
            pl_list.append(node)

    #
    insert_node = fe_list[-1]._get_next_node()  # before last node of FE section

    # create placeholder remapping
    pl_map: Dict[fx.Node, fx.Node] = {}
    pl_map[pl_list[0]] = gi.global_buffer_node  # type: ignore
    for i in range(num_fe_items):
        pl_map[pl_list[i + 1]] = fe_list[i].grad_tensor_node  # type: ignore

    update_node_user_count: Dict[fx.Node, str] = {}
    value_remap: Dict[fx.Node, fx.Node] = {}

    def remap_scatter_args(in_node: fx.Node) -> fx.Node:
        out_node = in_node
        if in_node in pl_map:
            out_node = pl_map[in_node]  # type: ignore
        elif in_node in value_remap:
            out_node = value_remap[in_node]

        update_node_user_count[out_node] = ""
        return out_node

    with gm.graph.inserting_before(insert_node):
        for innernode in scatter_sg.graph.nodes:
            if innernode.op in [OP.PLACEHOLDER, OP.OUTPUT]:
                continue
            value_remap[innernode] = gm.graph.node_copy(
                innernode, remap_scatter_args
            )

    # insert into main graph, just above last fe

    # force copies and waits to have a user
    # copies and waits do not have users by default, and will be
    # removed at recompile (can lead to lots of surprise/frustration)
    # # TODO this does not account for nodes beyond our own...remove/fix this

    _update_new_copy_nodes_users(value_remap)

    # also must update wait for the scatter section
    section_wait_node = scatter_list[-1].wait_node
    user = section_wait_node.args[0]  # type: ignore
    section_wait_node.users[user] = ""  # type: ignore
    wait_node_user_count = len(section_wait_node.users)  # type: ignore

    assert (
        wait_node_user_count > 0
    ), f"failed to update users for node {node.name}"

    # finally, need to update the graph TensorMetadata info (not a must, but ensures well formed graph)

    last_get_item_node = scatter_list[-1].wait_node.args[0]  # type: ignore
    tensor_meta = last_get_item_node.meta.get("tensor_meta", None)  # type: ignore
    assert (
        tensor_meta is not None
    ), f"failed to get tensor metadata for last getitem node {last_get_item_node=}"

    # replace with buffer metadata
    buffer_meta = gi.global_buffer_node.meta.get("tensor_meta", None)  # type: ignore

    new_tensor_meta = _update_node_tensor_metadata(
        last_get_item_node, new_shape=buffer_shape  # type: ignore
    )

    gm.recompile()


def _update_new_copy_nodes_users(value_remap: Dict[fx.Node, fx.Node]) -> None:
    """
    We have to manually update users for new copy nodes to ensure count > 0.
    This seems to be an fx bug, but for now we update or else fusion will get removed during graph linting
    """
    for subnode, node in value_remap.items():
        if node.name.startswith("copy"):
            _debug(
                f"426 copy or wait node pre user update len {len(node.users)}, {node.name=}, {node.users=}, {node.args=}"
            )
            # if len(node.users) == 0:
            user = node.args[0]
            node.users[user] = ""  # type: ignore
            node_user_len = len(node.users)
            assert node_user_len, f"failed to update users for node {node.name}"


def _update_node_tensor_metadata(
    node: fx.Node,
    new_shape: torch.Size,
    in_dtype: Optional[torch.dtype] = None,
    in_memory_format: Optional[torch.memory_format] = None,
) -> TensorMetadata:
    """Update a node's metadata to the the new shape, dtype and/or memory format"""
    curr = node.meta.get("tensor_meta")
    assert (
        curr is not None
    ), f"failed to obtain tensor meta data on node {node.name}"

    shape = curr.shape
    curr_dtype = curr.dtype
    requires_grad = curr.requires_grad
    stride = curr.stride

    curr_memory_format = curr.memory_format
    is_quantized = curr.is_quantized
    qparams = curr.qparams

    updated_dtype = in_dtype if in_dtype is not None else curr_dtype
    updated_memory_format = (
        in_memory_format if in_memory_format is not None else curr_memory_format
    )

    # tempt = torch.empty(new_shape)
    # new_shape = tempt.shape

    new_metadata = TensorMetadata(
        new_shape,
        updated_dtype,
        requires_grad,
        stride,
        updated_memory_format,
        is_quantized,
        qparams,
    )

    # update meta with new TensorMetadata
    saved_meta = node.meta.get("tensor_meta")
    node.meta["tensor_meta"] = new_metadata

    return new_metadata


def _finalize_output_node(
    gi: GraphInfo,
    gm: fx.GraphModule,
    fe_list: List[FusionElement],
    start: int,
    stop: int,
    new_output_args: List[fx.Node],
) -> None:
    """Reworks output node args to original grad tensors, replacing the wait_comms
    we update a copy of the output args, then finalized after all fusion is done."""

    # output_node = gi.output
    # new_output_args = []

    # curr_output_args = list(gi.output.args[0])
    replacement_mapping: Dict[fx.Node, fx.Node] = {}

    # map out all updated nodes in our list
    for item in fe_list:
        grad_node = item.grad_tensor_node
        wait_node = item.wait_node
        replacement_mapping[wait_node] = grad_node  # type: ignore

    # we have fused a subset, only update that subset within the larger output node args
    # TODO - this assumes that all gradient tensors are comm handled.
    for i in range(len(fe_list)):
        index = start + i
        curr_node = new_output_args[index]

        if curr_node is not None:
            assert curr_node.name.startswith(
                "wait"
            ), f"Non comm gradient output tensor incorrectly handled...needs fix. {new_output_args[start+i]}"
            new_output_args[start + i] = replacement_mapping[curr_node]

    _debug(f"537 - updated output args = {new_output_args}\n")


def _determine_peak_memory(gi: GraphInfo, fusion_policy: int) -> int:
    """
    Scans fe list to determine max memory required across all fusion instances.
    this result is used to allocate the global buffer for fusion, where we
    re-use a global buffer to avoid repeated allocations per fusion.
    """
    peak_memory = 0  # currently measured in numel
    curr_memory = 0
    curr_fe_index = 0
    for i, item in enumerate(gi.fe_list):  # type: ignore
        curr_fe_index += 1
        curr_memory += item.size  # type: ignore

        if curr_fe_index == fusion_policy:
            peak_memory = max(peak_memory, curr_memory)
            curr_fe_index = 0
            curr_memory = 0

    _debug(f"574, peak memory determined to be {peak_memory}")
    gi.peak_memory_required = peak_memory

    return peak_memory


def run_comm_fusion(gm: fx.GraphModule) -> None:
    """Main entry into remapping graph for all_reduce fusion.
    Modifications are in place to the graph.  Errors will result in stoppage
    to alert user rather than handling and returning error codes."""

    # first recompile to make sure we have coherent graph
    gm.recompile()

    # get our main graph info
    graph_info = GraphInfo()
    graph_info.update_info(gm)

    _debug(f"\n Start of fusion pass graph {gm.graph.print_tabular()}\n")

    # scan graph for all comm sections (fusion elements)
    fe_list = _scan_graph_for_fusion_elements(
        graph_info, gm, comm_type=CommType.allreduce
    )

    graph_info.num_starting_fe = len(fe_list)  # type: ignore
    graph_info.fe_list = fe_list

    # simple fusion policy where int = num buckets to fuse...start with 2,
    # meaning every 2 comms are fused into 1
    fusion_policy: int = 2

    # determine peak memory using fusion policy
    peak_memory_required = _determine_peak_memory(graph_info, fusion_policy)

    buffer_node = _insert_fusion_buffer_node(
        gm, peak_memory_required, graph_info
    )

    # Main process loop - iterate all fusion elements, apply fusion to subsets

    offset = 0
    count = 0

    start_output_args: List[fx.Node] = graph_info.output.args[0]  # type: ignore
    new_output_args: List[fx.Node] = list(start_output_args)  # type: ignore

    # ----------- main fusion loop ------------------------

    for index, item in enumerate(graph_info.fe_list):  # type: ignore
        count += 1
        if count == fusion_policy:
            start_index = offset
            stop_index = offset + count

            curr_fe_list = graph_info.fe_list[start_index:stop_index]  # type: ignore

            _copy_fe_to_buffer(graph_info, gm, curr_fe_list)

            _scatter_results_from_buffer(graph_info, gm, curr_fe_list)

            # switch wait_comms to output gradient nodes in output directly
            # fusion will have removed and reworked existing wait_comms

            _finalize_output_node(
                graph_info,
                gm,
                curr_fe_list,
                start_index,
                stop_index,
                new_output_args,
            )

            offset += count
            count = 0
    # update output with the updated args
    gm.graph.erase_node(graph_info.output)
    gm.graph.output(new_output_args)

    _debug(f"631, processed {index+1} fe items")

    _debug(f"656, updated output node args {new_output_args=}\n")

    # final review print
    graph_cleanup(gm)

    gm.recompile()
