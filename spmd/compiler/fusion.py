import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.fx.passes.shape_prop import TensorMetadata

from .graph_utils import (
    OP,
    get_all_nodes_of_type,
    create_graph_node_map,
    get_node_tensor_numel_shape,
    get_output_node,
    graph_cleanup,
    pretty_print_graph,
)
from .log_utils import rank0_debug

logger: logging.Logger = logging.getLogger(__name__)
_debug = partial(rank0_debug, logger)

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
    processed: bool = False
    size: int = 0
    shape: Optional[List[int]] = field(default_factory=lambda: [])  # type: ignore
    comm_type: Optional[CommType] = None
    node_list: Optional[List[fx.Node]] = field(default_factory=lambda: [])  # type: ignore
    prev_node: Optional[fx.Node] = None  # node that was before start of section
    output_name: str = ""
    comm_node: Optional[fx.Node] = None
    wait_node: Optional[fx.Node] = None
    grad_tensor_node: Optional[fx.Node] = None

    def _get_next_node(
        self,
    ):
        """get the next node after this FE section"""
        next_node = self.node_list[-1].next
        # _debug(f"57, next node name is {next_node.name}")
        assert (
            next_node is not None
        ), f"failed to get valid next node after {self.node_list[-1].name}"
        return next_node


@dataclass
class GraphInfo:
    """provides a home for global aspects of this graph.
    Currently tracks first and last node, len of the graph and
    the location and size of the global buffer node
    """

    len: int = 0
    num_starting_fe: int = 0
    fe_list: list = None
    peak_memory_required: int = 0
    global_buffer_node: Optional[fx.Node] = None
    global_buffer_size: int = 0
    tracing_buffer: Optional[torch.Tensor] = None
    first: Optional[fx.Node] = None
    output: Optional[fx.Node] = None

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
    gm: fx.GraphModule, buffer_size: Iterable[int], gi: GraphInfo = None
) -> fx.Node:
    """insert a torch.empty node for the global buffer.
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
    gm: fx.GraphModule,
    comm_type: CommType = CommType.allreduce,
) -> Optional[List[FusionElement]]:
    """scan entire graph for matching sections of CommTensor style expansions
    returns list of FusionElements that match CommType"""

    element_list = []

    fe_sequence = [
        # "clone",
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
                _debug(f"prev node = {fe.prev_node}")
                fe.next_node = node.next

                fe.output_name = node.name
                fe.wait_node = node
                fe.comm_node = curr_node_list[2]

                fe.grad_tensor_node = fe.comm_node.args[0][0]

                size, shape = get_node_tensor_numel_shape(fe.grad_tensor_node)  # type: ignore
                fe.size = size
                fe.shape = shape
                _debug(f"\nfe list size shape {size=}, {shape=}\n")

                element_list.append(fe)

            index = 0
            curr_node_list.clear()
            continue

    return element_list


def _copy_fe_to_buffer(
    gi: GraphInfo, gm: fx.GraphModule, in_fe_list: list[FusionElement]
) -> None:
    """first half of fusion - move desired items to buffer and create graph"""
    buffer_node = gi.global_buffer_node
    buffer_size = gi.global_buffer_size

    copy_list = in_fe_list

    num_fusion_elements = len(copy_list)

    def copy_to_buffer(buffer, tensor_list):
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
        buffer = buffer = torch.empty(buffer_size)

    tlist = []
    for item in copy_list:
        a = torch.zeros((item.shape[0], item.shape[1]))
        tlist.append(a)

    load_gm = make_fx(copy_to_buffer)(buffer, tlist)

    subnodemap = create_graph_node_map(load_gm)

    # update load loop to use main graph items
    fn_list = []
    pl_list = []
    for node in load_gm.graph.nodes:
        if node.op == OP.PLACEHOLDER:
            pl_list.append(node)
        elif node.op == OP.CALL_FUNCTION:
            fn_list.append(node)

    # create placeholder remapping
    pl_map = {}
    pl_map[pl_list[0]] = gi.global_buffer_node

    for i in range(num_fusion_elements):
        pl_map[pl_list[i + 1]] = in_fe_list[i].grad_tensor_node

    insert_node = in_fe_list[-1].comm_node

    _debug(f" f229 insert before comm node = {insert_node.name}\n")

    # TODO - this is debug only...remove
    #  verify first node
    for node in gm.graph.nodes:
        if node.name == insert_node.name:
            true_insert_node = node

    _debug(
        f" compare insert nodes for {insert_node.name}, true {id(true_insert_node)} vs fe list {id(insert_node)}"
    )

    def remap_copy_args(in_node: fx.Node) -> fx.Node:
        out_node = in_node
        if in_node in pl_map:
            out_node = pl_map[in_node]
            _debug(f"f249 remapped {in_node.name} to {out_node.name}")
        elif in_node in value_remap:
            out_node = value_remap[in_node]
            _debug(f"f252 remapped {in_node.name} to {out_node.name}")
        return out_node

    value_remap = {}
    with gm.graph.inserting_before(true_insert_node):
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

    buffer_comm_node = in_fe_list[-1].comm_node
    buffer_comm_node.update_arg(0, [buffer_node])

    _debug(f"f272 new comm node = {buffer_comm_node.args=}")

    _debug(f"f261 remapping\n {gm.graph.print_tabular()}")


def _build_buffer_comm_graph(gm, gi) -> fx.GraphModule:
    """have to make our own all_reduce and wait subgraph for buffer"""
    from torch.distributed._spmd.comm_tensor import CommTensor
    from torch.distributed.distributed_c10d import (
        ProcessGroup,
        ReduceOp,
        Work,
        _get_default_group,
        all_reduce,
    )

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
    _debug(f"\n315  process group = {pg}")
    tensor: torch.Tensor
    op: ReduceOp = ReduceOp.SUM  # type: ignore[assignment]
    async_op: bool = False

    # work_handle = all_reduce(grad_buffer, op=op, group=pg, async_op=async_op)

    _debug(f"303 \n{traced_add.graph.print_tabular()}\n")


def _scatter_results_from_buffer(
    gi: GraphInfo, gm: fx.GraphModule, fe_list: List[FusionElement]
) -> None:
    """after comm event with buffer, scatter results back to original fe grad tensors"""

    buffer_node = gi.global_buffer_node
    buffer_size = gi.global_buffer_size

    scatter_list = fe_list
    num_fe_items = len(scatter_list)

    def scatter_from_buffer(buffer, scatter_list):
        offset = 0
        for t in scatter_list:
            numel = t.numel()
            shaper = buffer[offset : offset + numel].view(t.shape)
            t.copy_(shaper)  # buffer[offset : offset + numel].view(t.shape() ))
            offset += numel
        return buffer

    # TODO - this is a dummy buffer that is never used,
    # it is just a proxy for the global buffer node.
    # consider simply saving and reusing a single dummy buffer

    buffer = torch.empty(buffer_size)
    buffer_shape = buffer.shape
    _debug(f"363, {buffer_shape=}\n")
    # _debug(f"buffer shape {buffer.shape}")
    tlist = []
    for item in scatter_list:
        shape = item.shape

        a = torch.zeros(item.shape[0], item.shape[1])

        tlist.append(a)  # clone().detach())

    scatter_sg = make_fx(scatter_from_buffer)(buffer, tlist)
    # _debug(f"f296 ==== {scatter_sg.graph.print_tabular()}\n")

    pl_list = []

    for node in scatter_sg.graph.nodes:
        if node.op == OP.PLACEHOLDER:
            pl_list.append(node)

    #
    insert_node = fe_list[-1]._get_next_node()  # before last node of FE section
    # _debug(f" f308 scatter to insert node after = {insert_node.name}\n")
    _debug(f"385 {insert_node=}\n")
    # verify first node
    # TODO - this is purely debug checking below..remove
    for node in gm.graph.nodes:
        if node.name == insert_node.name:
            true_insert_node = node

    _debug(
        f" compare insert nodes, true {id(true_insert_node)} vs fe list {id(insert_node)}"
    )

    # create placeholder remapping
    pl_map = {}
    pl_map[pl_list[0]] = gi.global_buffer_node
    for i in range(num_fe_items):
        pl_map[pl_list[i + 1]] = fe_list[i].grad_tensor_node

    update_node_user_count: Dict[str, None] = {}
    value_remap = {}

    def remap_scatter_args(in_node: fx.Node) -> fx.Node:
        out_node = in_node
        if in_node in pl_map:
            out_node = pl_map[in_node]
            _debug(f"f328 remapped {in_node.name} to {out_node.name}")
        elif in_node in value_remap:
            out_node = value_remap[in_node]
            _debug(f"f331 remapped {in_node.name} to {out_node.name}")

        update_node_user_count[out_node] = ""
        return out_node

    with gm.graph.inserting_before(true_insert_node):
        for innernode in scatter_sg.graph.nodes:
            if innernode.op in [OP.PLACEHOLDER, OP.OUTPUT]:
                continue
            value_remap[innernode] = gm.graph.node_copy(
                innernode, remap_scatter_args
            )

    # insert into main graph, just above last fe
    _debug(f"f341 = {value_remap=}\n")
    _debug(f"f341  ^^$$\n {gm.graph.print_tabular()}")

    _debug(f"421 - nodes we need to update users for {update_node_user_count=}")

    # force copies and waits to have a user
    # copies and waits do not have users by default, and will be
    # removed at recompile (can lead to lots of surprise/frustration)
    # # TODO this does not account for nodes beyond our own...remove/fix this
    # TODO - this is scanning entire graph every fusion...optimize

    _update_new_copy_nodes_users(value_remap)

    # also must update wait for the scatter section
    section_wait_node = scatter_list[-1].wait_node
    user = section_wait_node.args[0]
    section_wait_node.users[user] = ""
    _debug(f"445, section wait node user updated to {section_wait_node.users=}")
    assert (
        len(section_wait_node.users) > 0,
    ), f"failed to update users for node {node.name}"

    # finally, need to update the graph TensorMetadata info (not a must, but ensures well formed graph)

    last_get_item_node = scatter_list[-1].wait_node.args[0]
    _debug(f"447, {last_get_item_node.name=}")
    tensor_meta = last_get_item_node.meta.get("tensor_meta", None)
    assert (
        tensor_meta is not None
    ), f"failed to get tensor metadata for last getitem node {last_get_item_node=}"

    _debug(f"453, {tensor_meta=}\n")

    # replace with buffer metadata
    buffer_meta = gi.global_buffer_node.meta.get("tensor_meta", None)
    _debug(f"459, {buffer_meta=}")

    new_tensor_meta = _update_node_tensor_metadata(
        last_get_item_node, new_shape=buffer_shape
    )
    _debug(f"466, new get item meta = {new_tensor_meta=}")

    gm.recompile()

    # TODO - another patch to update meta data..hardcoded for first test.
    # update to direct
    for node in gm.graph.nodes:
        if node.name.startswith("getitem_3"):
            _debug(
                f"377 get item node {node.name=}, {node.users=}, {node.args=}"
            )
            tdata = node.meta.get("tensor_meta")

            _debug(f"379 meta type = {type(tdata)}, data = {tdata}")
            # tdata["shape"] = (200,)

            # node.users[user] = ""
            # _debug(f"369 copy node {node.name=}, {node.users=}, {node.args=}")
    gm.recompile()

    _debug(f"446 {print(gm.graph)}")


def _update_new_copy_nodes_users(value_remap):
    """
    we have to manually update users for new copy nodes to ensure count > 0.
    This seems to be an fx bug, but for now we update or else fusion will get removed during graph linting
    """
    for subnode, node in value_remap.items():
        if node.name.startswith("copy"):
            _debug(
                f"426 copy or wait node pre user update len {len(node.users)}, {node.name=}, {node.users=}, {node.args=}"
            )
            # if len(node.users) == 0:
            user = node.args[0]
            node.users[user] = ""
            assert (
                len(node.users) > 0,
            ), f"failed to update users for node {node.name}"


def _update_node_tensor_metadata(
    node, new_shape: tuple, dtype=None, memory_format=None
):
    """update a node's metadata to the the new shape, dtype and/or memory format"""
    curr = node.meta.get("tensor_meta")
    assert (
        curr is not None
    ), f"failed to obtain tensor meta data on node {node.name}"

    _debug(f"f551, starting meta = {curr=}")

    shape = curr.shape
    dtype = curr.dtype
    requires_grad = curr.requires_grad
    stride = curr.stride

    memory_format = curr.memory_format
    is_quantized = curr.is_quantized
    qparams = curr.qparams

    # tempt = torch.empty(new_shape)
    # new_shape = tempt.shape

    new_metadata = TensorMetadata(
        new_shape,
        dtype,
        requires_grad,
        stride,
        memory_format,
        is_quantized,
        qparams,
    )

    _debug(
        f"541, new metadata = {new_metadata} and shape type = {type(new_metadata.shape)}"
    )

    # update meta with new TensorMetadata
    saved_meta = node.meta.get("tensor_meta")

    try:
        node.meta["tensor_meta"] = new_metadata
    except:
        print(f"FAILED to update meta")

    return new_metadata


def _finalize_output_node(gi, gm, fe_list, start, stop, new_output_args):
    """reworks output node args to original grad tensors, replacing the wait_comms
    we update a copy of the output args, then finalized after all fusion is done."""

    # output_node = gi.output
    # new_output_args = []

    # curr_output_args = list(gi.output.args[0])
    replacement_mapping: Dict[fx.Node, fx.Node] = {}

    # map out all updated nodes in our list
    for item in fe_list:
        grad_node = item.grad_tensor_node
        wait_node = item.wait_node
        replacement_mapping[wait_node] = grad_node
        # new_output_args.append(grad_node)
    # new_output_args.append(None)
    _debug(
        f"583, ready to update main output with these replacements: {replacement_mapping}\n"
    )

    _debug(f"\n 570 - new output args = {new_output_args}\n ")
    for item in new_output_args:
        if item is not None:
            _debug(f"572 - node {item.name}, type {type(item)}\n")
        else:
            _debug("572, None,  ending node found")

    # we have fused a subset, only update that subset within the larger output node args
    # TODO - this assumes that all gradient tensors are comm handled.
    for i in range(len(fe_list)):
        index = start + i
        curr_node = new_output_args[index]
        _debug(f"582 - node is {curr_node}")
        if curr_node is not None:
            assert curr_node.name.startswith(
                "wait"
            ), f"Non comm gradient output tensor incorrectly handled...needs fix. {new_output_args[start+i]}"
            new_output_args[start + i] = replacement_mapping[curr_node]
            _debug(
                f"605, replacing {curr_node} with {replacement_mapping[curr_node]}"
            )

    _debug(f"577 - updated output args = {new_output_args}\n")


def _determine_peak_memory(gi: GraphInfo, fusion_policy: int) -> int:
    """
    scans fe list to determine max memory required across all fusion instances.
    this result is used to allocate the global buffer for fusion, where we
    re-use a global buffer to avoid repeated allocations per fusion.
    """
    peak_memory = 0  # currently measured in numel
    curr_memory = 0
    fast_index = 0
    for i, item in enumerate(gi.fe_list):
        fast_index += 1
        curr_memory += item.size

        if fast_index == fusion_policy:
            peak_memory = max(peak_memory, curr_memory)
            fast_index = 0
            curr_memory = 0

    _debug(f"574, peak memory determined to be {peak_memory}")
    gi.peak_memory_required = peak_memory

    return peak_memory


def run_comm_fusion(gm: fx.GraphModule) -> bool:
    """main entry into remapping graph for all_reduce fusion"""

    result = False

    # first recompile to make sure we have coherent graph
    gm.recompile()

    # get our main graph info
    gi = GraphInfo()
    gi.update_info(gm)

    _debug(f"\n Start of fusion pass graph {gm.graph.print_tabular()}\n")

    # scan graph for all comm sections (fusion elements)
    fe_list = _scan_graph_for_fusion_elements(gm, comm_type=CommType.allreduce)

    gi.num_starting_fe = len(fe_list)
    gi.fe_list = fe_list

    # simple fusion policy where int = num buckets to fuse...start with 2,
    # meaning every 2 comms are fused into 1
    fusion_policy: int = 2

    # compute optimal buffer size here....
    # this will be based on either max bucket or bucket schedule from external optimizer
    # or we can scan the graph and auto-compute max size for any single fusion
    # TODO

    # determine peak memory using fusion policy
    peak_memory_required = _determine_peak_memory(gi, fusion_policy)

    buffer_node = _insert_fusion_buffer_node(gm, peak_memory_required, gi)

    # Main process loop - iterate all fusion elements, apply fusion to subsets

    offset = 0
    count = 0
    # new_output_args=[]
    start_output_args = gi.output.args[0]
    new_output_args = list(start_output_args)
    _debug(f"671, starting output args {start_output_args}\n")

    # ----------- main fusion loop ------------------------

    for index, item in enumerate(gi.fe_list):
        count += 1
        if count == fusion_policy:
            start_index = offset
            stop_index = offset + count

            curr_fe_list = gi.fe_list[start_index:stop_index]

            _copy_fe_to_buffer(gi, gm, curr_fe_list)

            _scatter_results_from_buffer(gi, gm, curr_fe_list)

            # switch wait_comms to output gradient nodes in output directly
            # fusion will have removed and reworked existing wait_comms
            # TODO - this will break atm for dynamic fusion...rework for unlimited fusion case.
            _finalize_output_node(
                gi, gm, curr_fe_list, start_index, stop_index, new_output_args
            )

            offset += count
            count = 0
    # update output with the updated args
    gm.graph.erase_node(gi.output)
    gm.graph.output(new_output_args)

    _debug(f"631, processed {index+1} fe items")

    # final verification of output node - # TODO remove as this is debugging util
    for node in reversed(gm.graph.nodes):
        if node.op == OP.OUTPUT:
            new_output = node
            break
    _debug(f"703, updated output node args {new_output.args=}\n")

    _debug(f"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    # final review print
    graph_cleanup(gm)

    _debug(f"\n Start of meta pass graph {gm.graph.print_tabular()}\n")

    # TODO - remove this.  This is purely a final check to verify all meta data
    # related to fusion has been updated to use the buffer size rather than original.

    get_nodes = get_all_nodes_of_type(
        gm, OP.CALL_FUNCTION, starts_with="get", require_meta=True
    )

    _debug(f"\672 ++++++++++++++++ \n{get_nodes=}\n")
    # ---- end final meta tensors debug review -----------

    _debug(f"577, global buffer size = {gi.global_buffer_size}")

    result = True  # TODO - make this mean something

    gm.recompile()
    return gm
