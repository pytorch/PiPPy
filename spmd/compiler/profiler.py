# type: ignore
# pyre-ignore-all-errors
import logging
from dataclasses import fields
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from functorch.compile import aot_module, make_boxed_func
from torch import fx
from torch.autograd.profiler_util import EventList
from torch.fx.node import map_arg
from torch.profiler import profile, ProfilerActivity, record_function, schedule

from .graph_utils import OP
from .profiler_utils import (
    BiDict,
    get_tensor_stats,
    GraphType,
    IntermediateNodeInfo,
    NodeInfo,
    ProfileMode,
    TensorStatus,
)

MEM_LIMIT = 0

if torch.cuda.is_available():
    MEM_LIMIT = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory


class GraphProfiler(fx.Interpreter):
    r"""The main GraphProfiler class that extends the fx.Interpreter and runs
    the input graph module node by node, collecting profiling information for
    each one of them.
    Args:
    gm (fx.GraphModule): The fx graphmodule to initialize the
                                GraphProfiler.
    gtype (GraphType): The type of fx graph module (forward/backward)
    fwd_profiler (GraphProfiler): To intialize the backward
                                profiler, an instance of forward profiler is
                                required. It is used to create a mapping from
                                nodes of forward graph to the backward graph,
                                merge the profiling information from forward
                                and backward graphs into a single dictionary
                                and use the attributes of the fwd_profiler.
    fwd_num_outs (int): The number of outputs in the original dynamo graph used
                        to generate the aot_graphs.
    sync (bool): Flag to indicate whether the cuda stream should be
                synchronized for each node operation.
    profile_mode (str): The Graph Profiler provides three profiling
                        modes,``default``, ``memory`` and ``swap``.
    """
    prefix_str: str
    sync: bool
    node_info: Dict[fx.Node, NodeInfo]
    fwd_intermediate_nodes: List[fx.Node]
    fwd_intermediate_nodes_flags: List[bool]
    fwd_num_outs: int
    profile_mode: str

    def __init__(
        self,
        gm: fx.GraphModule,
        gtype: GraphType,
        fwd_profiler: Optional["GraphProfiler"] = None,
        fwd_num_outs: int = 1,
        sync: bool = False,
        profile_mode: str = "default",
    ):
        super().__init__(gm, True)
        self.gm = gm
        print("Current Device: ", torch.cuda.current_device())
        self.gtype: GraphType = gtype
        torch.cuda.reset_peak_memory_stats()
        if self.gtype == GraphType.BACKWARD:
            logging.info("Initializing Backward Profiler")
            assert fwd_profiler is not None
            self.prefix_str = "profiler_bw"
            self.sync = fwd_profiler.sync
            self.node_info = fwd_profiler.node_info
            self.fwd_intermediate_nodes = fwd_profiler.intermediate_nodes
            self.fwd_intermediate_nodes_flags = (
                fwd_profiler.fwd_intermediate_nodes_flags
            )
            self.fwd_num_outs = fwd_profiler.fwd_num_outs
            self.profile_mode = fwd_profiler.profile_mode
        else:
            logging.info("Initializing Forward Profiler")
            self.prefix_str = "profiler_fw"
            self.sync = sync
            self.node_info = {}
            self.fwd_intermediate_nodes = []
            self.fwd_intermediate_nodes_flags = []
            self.fwd_num_outs = fwd_num_outs
            self.profile_mode = profile_mode

        self.total_runtime_sec: List[float] = []
        self.attr_map: Dict[fx.Node, Any] = {}
        self.node_active_mem: Dict[fx.Node, List[int]] = {}
        self.node_peak_mem: Dict[fx.Node, List[int]] = {}
        self.runtimes_sec: Dict[fx.Node, float] = {}
        self.swaptimes_sec: Dict[fx.Node, float] = {}
        self.node_cuda_time: Dict[fx.Node, float] = {}
        self.node_cpu_time: Dict[fx.Node, float] = {}
        self.node_cuda_swaptime: Dict[fx.Node, float] = {}
        self.node_cpu_swaptime: Dict[fx.Node, float] = {}
        self.intermediate_nodes: List[fx.Node] = []
        self.torch_profiler: Optional[torch.profiler.profile] = None
        self.prev_runtime: float = 0
        self.needs_summary: bool = True
        self.env = {}

        # Can define any variables that you need to measure the runtime events
        # at the Node level

        # If graph type is forward then find the last use of the intermediate
        # tensors during the forward pass The output node contains all the
        # tensors live at the end of forward pass Each node generates an
        # intermediate tensor, check if that tensor is active at end If yes, add
        # it to intermediate tensor list, find its last use node
        if gtype == GraphType.FORWARD:
            self._init_fwd_profiler()
        elif gtype == GraphType.BACKWARD:
            self._init_bwd_profiler()

    def _init_fwd_profiler(self) -> None:
        # For the intermediate nodes obtain their last use in the forward
        # pass excluding the output node
        node_to_last_forward_use: Dict[fx.Node, fx.Node] = {}
        # stores the last forward uses
        self.user_to_last_forward_uses: Dict[fx.Node, List[fx.Node]] = {}

        for node in self.module.graph.nodes:
            if node.op != OP.OUTPUT:
                continue
            # get all the arguments form the output node these are all
            # the nodes that are live at the end of forward pass
            op_nodes = node.all_input_nodes[self.fwd_num_outs :]
            for n in op_nodes:
                # We want to exclude the placeholder nodes since they
                # represent the model parameters
                if n.op != OP.PLACEHOLDER:
                    ip_nodes: List[fx.Node] = n.all_input_nodes
                    is_placeholder = [
                        True if (inode.op == OP.PLACEHOLDER) else False
                        for inode in ip_nodes
                    ]
                    if all(is_placeholder):
                        self.fwd_intermediate_nodes_flags.append(False)
                    else:
                        self.intermediate_nodes.append(n)
                        self.fwd_intermediate_nodes_flags.append(True)
                else:
                    self.fwd_intermediate_nodes_flags.append(False)
        rank = 0
        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            n_info: NodeInfo = NodeInfo()
            if node in self.intermediate_nodes:
                n_info = IntermediateNodeInfo()
                # NOTE:This condition is especially for the node that is
                # directly used in the output after generation but not a
                # part of the output
                users_count = len(node.users)
                if users_count == 1:
                    user = list(u for u in node.users.keys())[0]
                    assert type(user) == fx.Node
                    if user.op == OP.OUTPUT:
                        self.user_to_last_forward_uses.setdefault(
                            user, []
                        ).append(node)
                        n_info.last_forward_access = user
            n_info.rank = rank
            rank += 1
            n_info.gtype = GraphType.FORWARD
            self.node_info[node] = n_info

        def register_last_forward_uses(n: fx.Node, user: fx.Node):
            if (
                n not in node_to_last_forward_use
                and n in self.intermediate_nodes
            ):
                node_to_last_forward_use[n] = user
                self.node_info[n].last_forward_access = user
                self.user_to_last_forward_uses.setdefault(user, []).append(n)

        # We traverse the nodes in a reverse order and find first use of the
        # intermediate tensor in its forward pass
        for node in reversed(self.module.graph.nodes):
            # Exclude the output node
            if node.op == OP.OUTPUT:
                continue
            map_arg(node.args, lambda n: register_last_forward_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_forward_uses(n, node))

        # For the parameter nodes obtain their first use in the forward pass
        # excluding the output node
        node_to_first_forward_use: Dict[fx.Node, fx.Node] = {}
        # stores the first forward uses
        self.user_to_first_forward_uses: Dict[fx.Node, List[fx.Node]] = {}

        # registering first forward uses for parameters
        def register_first_forward_uses(n: fx.Node, user: fx.Node):
            if n not in node_to_first_forward_use and n.op == OP.PLACEHOLDER:
                node_to_first_forward_use[n] = user
                self.node_info.setdefault(
                    n, NodeInfo()
                ).first_forward_access = user
                self.user_to_first_forward_uses.setdefault(user, []).append(n)

        for node in self.module.graph.nodes:
            # Exclude the output node
            if node.op == OP.OUTPUT:
                continue
            map_arg(node.args, lambda n: register_first_forward_uses(n, node))
            map_arg(node.kwargs, lambda n: register_first_forward_uses(n, node))

    def _init_bwd_profiler(self) -> None:
        # populate the intermediate nodes for the backward pass as well
        fwd_intermediate_nodes_iterator: Iterator[Any] = iter(
            self.fwd_intermediate_nodes
        )
        self.fwd_bwd_intermediate: BiDict[fx.Node, fx.Node] = BiDict()
        placeholders = [
            node
            for node in self.module.graph.nodes
            if node.op == OP.PLACEHOLDER
        ]
        placeholders = placeholders[: (len(placeholders) - self.fwd_num_outs)]
        assert len(placeholders) == len(self.fwd_intermediate_nodes_flags)

        for node, is_intermediate in zip(
            placeholders, self.fwd_intermediate_nodes_flags
        ):
            if is_intermediate:
                self.intermediate_nodes.append(node)
                fwd_node = next(fwd_intermediate_nodes_iterator)
                self.fwd_bwd_intermediate[fwd_node] = node

        rank = 0
        n_info: NodeInfo
        for node in self.module.graph.nodes:
            if node.op != OP.PLACEHOLDER:
                n_info = NodeInfo()
                n_info.rank = rank
                n_info.gtype = GraphType.BACKWARD
                rank += 1
                self.node_info[node] = n_info

        for node in self.module.graph.nodes:
            last_uses: List[fx.Node] = self.user_to_last_uses.get(node, [])
            for lunode in last_uses:
                if lunode in self.intermediate_nodes:
                    f_node = self.fwd_bwd_intermediate.inverse.get(lunode)[0]
                    n_info = self.node_info[f_node]
                    n_info.last_back_access = node

        # must have input list of intermediate nodes
        node_to_first_backward_use: Dict[fx.Node, fx.Node] = {}
        self.user_to_first_backward_uses: Dict[fx.Node, List[fx.Node]] = {}

        def register_first_backward_use(n: fx.Node, user: fx.Node):
            if (
                n not in node_to_first_backward_use
                and n in self.intermediate_nodes
            ):
                node_to_first_backward_use[n] = user
                f_node = self.fwd_bwd_intermediate.inverse.get(n)[0]
                assert isinstance(f_node, fx.Node)
                n_info = self.node_info[f_node]
                n_info.first_back_access = user
                self.user_to_first_backward_uses.setdefault(user, []).append(n)

        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            map_arg(node.args, lambda n: register_first_backward_use(n, node))
            map_arg(node.kwargs, lambda n: register_first_backward_use(n, node))

    def meta_run(self, *args) -> Any:
        args_iter = iter(args)
        for n in self.module.graph.nodes:
            if n.op == OP.PLACEHOLDER:
                self.env[n] = next(args_iter)
        args = None
        return self.run([])

    def run(self, *args) -> Any:
        if self.gtype == GraphType.FORWARD:
            self.param_memory: int = torch.cuda.memory_allocated()
        return_val = super().run(*args, initial_env=self.env)
        args = None
        if self.gtype == GraphType.BACKWARD:
            torch.cuda.synchronize()
            self.param_grad_memory: int = torch.cuda.memory_allocated()
        self.env = {}
        return return_val

    def _swap_out_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be offloaded
        # 2) Retrieve their CPU reference (if none allocate a CPU tensor in
        #    pinned memory)
        # 3) Copy the tensor to the CPU, add the CPU tensor to the Interpreter
        #    environment
        # 4) Delete the GPU tensor
        nodes_to_offload = self.user_to_last_forward_uses.get(node, [])
        for o_node in nodes_to_offload:
            cpu_ref: torch.Tensor = self.node_info[o_node].cpu_ref
            tensor = self.env[o_node]
            assert isinstance(tensor, torch.Tensor)
            if cpu_ref is None:
                cpu_ref = torch.zeros(
                    tensor.size(), dtype=tensor.dtype, layout=tensor.layout
                ).pin_memory()
            assert cpu_ref.is_pinned
            with record_function(f"{self.prefix_str}_{n.name}_swap"):
                cpu_ref = cpu_ref.copy_(tensor, False)
            if self.sync:
                torch.cuda.synchronize()
            self.node_info[o_node].status = TensorStatus.cpu
            self.node_info[o_node].cpu_ref = cpu_ref
            self.env[o_node] = cpu_ref
            del tensor
            tensor = None
            cpu_ref = None

    def _swap_in_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be prefetched
        # 2) Retrieve their CPU reference (assert if it resides in pinned
        #    memory)
        # 3) Copy the tensor to GPU memory and add it to the Interpreter
        #    environment
        # 4) Update the state of intermediate tensor in NodeInfo
        nodes_to_fetch = self.user_to_first_backward_uses.get(node, [])
        for p_node in nodes_to_fetch:
            f_node = self.fwd_bwd_intermediate.inverse.get(p_node)[0]
            assert isinstance(f_node, fx.Node)
            n_info = cast(IntermediateNodeInfo, self.node_info[f_node])
            n_info.status = TensorStatus.gpu
            cpu_ref = cast(torch.Tensor, n_info.cpu_ref)
            assert isinstance(cpu_ref, torch.Tensor) and cpu_ref.is_pinned
            with record_function(f"{self.prefix_str}_{f_node.name}_swap"):
                tensor = cpu_ref.to(
                    device=torch.cuda.current_device(),
                    memory_format=torch.preserve_format,
                )
            self.env[p_node] = tensor.contiguous()
            if self.sync:
                torch.cuda.synchronize()

    def run_node(self, node: fx.Node) -> Any:
        if node.op == OP.PLACEHOLDER:
            return super().run_node(node)

        # Preftech the tensors that have been offloaded and have their first uses.
        if (
            self.profile_mode == ProfileMode.swap
            and self.gtype == GraphType.BACKWARD
        ):
            self._swap_in_node(node)
        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

        with record_function(f"{self.prefix_str}_{node.name}"):
            return_val = super().run_node(node)
        if self.sync:
            torch.cuda.synchronize()

        if node.op == OP.GET_ATTR:
            self.attr_map[node] = return_val
        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            mem_stats = torch.cuda.memory_stats()
            self.node_peak_mem.setdefault(node, [])
            self.node_peak_mem[node].append(mem_stats["active_bytes.all.peak"])
            self.node_active_mem.setdefault(node, [])
            self.node_active_mem[node].append(
                mem_stats["active_bytes.all.current"]
            )
            if (
                self.gtype == GraphType.FORWARD
                and node in self.intermediate_nodes
            ):
                int_n_info = cast(IntermediateNodeInfo, self.node_info[node])
                assert isinstance(return_val, torch.Tensor)
                (
                    int_n_info.size,
                    int_n_info.numel,
                    int_n_info.memory_size,
                ) = get_tensor_stats(return_val)

        # Offload the tensors that have last uses at this node during forward pass.
        if (
            self.profile_mode == ProfileMode.swap
            and self.gtype == GraphType.FORWARD
        ):
            self._swap_out_node(node)

        return return_val

    def reset_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        self.total_runtime_sec: List[float] = []
        self.node_active_mem: Dict[fx.Node, List[Any]] = {}
        self.node_peak_mem: Dict[fx.Node, List[Any]] = {}
        self.runtimes_sec: Dict[fx.Node, List[float]] = {}
        self.swaptimes_sec: Dict[fx.Node, List[float]] = {}

    def get_idle_times(self) -> None:
        for i_node in self.intermediate_nodes:
            if self.gtype == GraphType.FORWARD:
                fn_info: IntermediateNodeInfo = self.node_info[i_node]
                last_use = fn_info.last_forward_access
                fn_info.idle_time = self.total_runtime - (
                    self.node_info[last_use].cumulative_run_time
                    + fn_info.swap_time
                )
            else:
                f_node = self.fwd_bwd_intermediate.inverse.get(i_node)[0]
                fn_info: IntermediateNodeInfo = self.node_info[f_node]
                first_use = fn_info.first_back_access
                fn_info.idle_time += self.node_info[
                    first_use
                ].cumulative_run_time - (
                    self.node_info[first_use].run_time + fn_info.swap_time
                )

    def get_peakmem_usage(self) -> None:
        if self.profile_mode == ProfileMode.swap:
            intermediate_mem = 0
            if self.gtype == GraphType.BACKWARD:
                for i_node in self.intermediate_nodes:
                    f_node = self.fwd_bwd_intermediate.inverse.get(i_node)[0]
                    fn_info: IntermediateNodeInfo = self.node_info[f_node]
                    intermediate_mem += fn_info.memory_size

            self.peak_start = None
            self.peak_end = None
            peak_interval: bool = False
            peak_end_reset: bool = False
            self.max_peak_mem = 0
            self.min_peak_mem = 0
            for node in self.module.graph.nodes:
                if node.op == OP.PLACEHOLDER:
                    continue
                if self.gtype == GraphType.BACKWARD:
                    nodes_to_prefetch = self.user_to_first_backward_uses.get(
                        node, None
                    )
                    if nodes_to_prefetch is not None:
                        for p_node in nodes_to_prefetch:
                            f_node = self.fwd_bwd_intermediate.inverse.get(
                                p_node
                            )[0]
                            intermediate_mem -= self.node_info[
                                f_node
                            ].memory_size
                min_peak_mem = self.node_info[node].peak_mem
                peak_mem = min_peak_mem + intermediate_mem
                if peak_mem > MEM_LIMIT:
                    peak_interval = True
                    peak_end_reset = True
                    if self.peak_start is None:
                        self.peak_start = node
                else:
                    peak_interval = False
                    if peak_end_reset:
                        self.peak_end = node
                        peak_end_reset = False

                self.node_info[node].in_peak_interval = peak_interval
                self.node_info[node].total_peak_mem = peak_mem
                self.max_peak_mem = max(self.max_peak_mem, peak_mem)
                self.min_peak_mem = max(self.min_peak_mem, min_peak_mem)

                if self.gtype == GraphType.FORWARD:
                    nodes_to_offload = self.user_to_last_forward_uses.get(
                        node, None
                    )
                    if nodes_to_offload is not None:
                        for o_node in nodes_to_offload:
                            intermediate_mem += self.node_info[
                                o_node
                            ].memory_size
        else:
            peak_mem_usages = [
                self.node_info[n].peak_mem
                for n in self.module.graph.nodes
                if n.op != OP.PLACEHOLDER
            ]
            self.max_peak_mem = max(peak_mem_usages)
            self.min_peak_mem = min(peak_mem_usages)
            self.peak_start = None
            self.peak_end = None

    def get_node_runtimes(self):
        assert self.torch_profiler is not None
        event_list_avg: EventList = self.torch_profiler.key_averages()
        event_dict: Dict[str, Tuple[float, float]] = {}
        prefix = self.prefix_str
        for e in event_list_avg:
            if prefix in e.key:
                event_dict[e.key] = (e.cuda_time, e.cpu_time)
        for n in self.module.graph.nodes:
            if n.op != OP.PLACEHOLDER:
                cuda_time, cpu_time = event_dict[f"{self.prefix_str}_{n.name}"]
                self.node_cuda_time[n] = cuda_time / 1000.0
                self.node_cpu_time[n] = cpu_time / 1000.0
                self.runtimes_sec[n] = max(cpu_time, cuda_time) / 1000.0
        if (
            self.profile_mode == ProfileMode.swap
            and self.gtype == GraphType.FORWARD
        ):
            for int_n in self.intermediate_nodes:
                cuda_time, cpu_time = event_dict[
                    f"{self.prefix_str}_{int_n.name}_swap"
                ]
                self.node_cuda_swaptime[int_n] = cuda_time / 1000.0
                self.node_cpu_swaptime[int_n] = cpu_time / 1000.0
                self.swaptimes_sec[int_n] = max(cpu_time, cuda_time) / 1000.0

    def summarize(self) -> None:
        if not self.needs_summary:
            return

        self.get_node_runtimes()
        self.total_runtime = 0

        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue

            n_info: NodeInfo = self.node_info.setdefault(node, NodeInfo())
            n_info.run_time = self.runtimes_sec.get(node, 1.0)
            n_info.cuda_time = self.node_cuda_time.get(node, 1.0)
            n_info.cpu_time = self.node_cpu_time.get(node, 1.0)
            n_info.exe_time = n_info.run_time
            self.total_runtime += n_info.run_time
            n_info.cumulative_run_time = self.total_runtime

            if self.profile_mode not in [ProfileMode.swap, ProfileMode.memory]:
                continue
            n_info.peak_mem = max(self.node_peak_mem.setdefault(node, [0]))
            n_info.active_mem = max(self.node_active_mem.setdefault(node, [0]))

            if (
                node in self.intermediate_nodes
                and self.profile_mode == ProfileMode.swap
            ):
                n_info: IntermediateNodeInfo = self.node_info[node]
                n_info.swap_time = self.swaptimes_sec[node]

        self.get_idle_times()
        self.get_peakmem_usage()

    def print_summary(self) -> str:
        try:
            import tabulate
        except ImportError:
            return "No tabulate module is found, skip printing summary."

        node_summaries: List[List[Any]] = []
        mean_total_runtime = self.total_runtime
        logging.info(f"Execution Time (ms): {self.total_runtime}")
        logging.info(f"Max Peak Mem Usage (B): {self.max_peak_mem}")

        headers: List[str] = [
            "Target",
            "Op",
            "Average runtime (ms)",
            "Pct total runtime",
            "CUDA time(ms)",
            "CPU time(ms)",
        ]
        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            headers.extend(
                [
                    "Mem Active (B)",
                    "Mem Peak Active(B)",
                    "Tensor Size(B)",
                ]
            )
        if self.profile_mode == ProfileMode.swap:
            print(
                "Peak Interval : ",
                str(self.peak_start),
                " - ",
                str(self.peak_end),
            )
            headers.extend(
                [
                    "Swap Time (ms)",
                    "Idle_time(ms)",
                    "Simulated Peak Active(B)",
                ]
            )
        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            n_info: NodeInfo = self.node_info[node]
            pct_total = n_info.run_time / mean_total_runtime * 100
            val_list = [
                node.target,
                str(node),
                n_info.run_time,
                pct_total,
                n_info.cuda_time,
                n_info.cpu_time,
            ]
            if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
                val_list.extend([n_info.active_mem, n_info.peak_mem])
            if node in self.intermediate_nodes:
                n_info: IntNodeInfo = n_info
                if self.profile_mode == ProfileMode.memory:
                    val_list.append(n_info.memory_size)
                if self.profile_mode == ProfileMode.swap:
                    val_list.extend(
                        [n_info.memory_size, n_info.swap_time, n_info.idle_time]
                    )
            else:
                if self.profile_mode == ProfileMode.memory:
                    val_list.append("")
                if self.profile_mode == ProfileMode.swap:
                    val_list.extend(["", "", ""])
            if self.profile_mode == ProfileMode.swap:
                val_list.append(n_info.total_peak_mem)
            node_summaries.append(val_list)
        return tabulate.tabulate(node_summaries, headers=headers)


class ProfilerEngine:
    r"""Obtain the forward pass and backward pass of the provided nn.Module
    and profile them. It provides the run function which takes an optional
    argument for running warm-up iterations before doing the actual profiling.

    Args: model (nn.Module): a local model instance of nn.Module.
    forward_loss(Callable): a function that takes and nn.module and input.
                            It calls the model with the provided inputs and
                            calculates and returns the loss.
    optimizer (Optional[optim.Optimizer]) :
    example_inputs (Any): The example inputs will be passed to the forward_loss
                        function to obtain the forward pass and loss of the
                        model.
    profile_mode (str): The Graph Profiler provides three profiling modes,
                        ``default``,``memory`` and ``swap``.
                default: Measure the per node run-time, marks the intermediate
                        nodes (activations saved from forward pass and needed in
                        backward pass), registers their last use in the forward
                        pass and first use in the backward pass, measures this
                        idle time and, marks the irst use of the model parameter
                        nodes in the forward pass.
                memory: All of the above plus active memory usage,
                        peak memory usage and intermediate (activation) memory.
                swap:   All the of the above plus profiles in a low memory
                        mode, pushing all of the activations to the CPU
                        memory during the forward pass and fetches them
                        back when they are needed in the backward pass.
                        It measures the time to swap each of the intermediate
                        tensors (activations) to CPU memory, back and forth.
                        Allows profiling graphs way larger than GPU memory.
    """

    def __init__(
        self,
        module: nn.Module,
        forward_loss: Callable,
        optimizer: Optional[torch.optim.Optimizer] = None,
        profile_mode: str = "default",
        warm_up_iters: int = 0,
        profile_iters: int = 1,
        dist_fwd_gm: Optional[fx.GraphModule] = None,
        dist_bwd_gm: Optional[fx.GraphModule] = None,
    ) -> None:
        self.module: nn.Module = module
        self.forward_loss = forward_loss
        # To account for optimizer memory usage, one step is required
        # because optimizer state is lazily initialized. The original
        # implementation does not consider this case. Make optimizer
        # as optional for now.
        self.optimizer = optimizer
        self.profile_mode = profile_mode
        self.profilers: Dict[int, Dict[GraphType, GraphProfiler]] = {0: {}}
        self.warm_up_iters = warm_up_iters
        self.profile_iters = profile_iters
        self.dist_fwd_gm = dist_fwd_gm
        self.dist_bwd_gm = dist_bwd_gm

    def _iter_all_profilers(self) -> Iterator[GraphProfiler]:
        for prof_dict in self.profilers.values():
            fwd_profiler = prof_dict.get(GraphType.FORWARD, None)
            if fwd_profiler is not None:
                yield fwd_profiler
            bwd_profiler = prof_dict.get(GraphType.BACKWARD, None)
            if bwd_profiler is not None:
                yield bwd_profiler

    def _aot_compile_fwd(
        self, gm: fx.GraphModule, inps: List[torch.Tensor]
    ) -> fx.GraphModule:
        # Wraps the forward compiler for the aot_module.
        # 1) It initializes the forward graph profiler.
        # 2) Stores the reference of the profiler.
        # 3) Plugs-in the profiler's run method as a callable to the forward pass.
        logging.info("Compiling Forward Graph")

        if self.dist_fwd_gm:
            gm = self.dist_fwd_gm

        output_count = 0
        for node in gm.graph.nodes:
            if node.op == OP.OUTPUT:
                for _ in node.args[0]:
                    output_count += 1

        fwd_profiler: GraphProfiler = GraphProfiler(
            gm,
            GraphType.FORWARD,
            fwd_num_outs=output_count,
            sync=False,
            profile_mode=self.profile_mode,
        )
        self.profilers[0][GraphType.FORWARD] = fwd_profiler

        return make_boxed_func(fwd_profiler.run)

    def _aot_compile_bwd(
        self, gm: fx.GraphModule, inps: List[torch.Tensor]
    ) -> fx.GraphModule:
        # Wraps the backward compiler for the aot_module.
        # 1) It initializes the backward graph profiler using the corresponding
        #    forward profiler.
        # 2) Stores the reference of the profiler.
        # 3) Plugs-in the profiler's ``meta_run`` method as a callable to the
        #    backward pass.
        # NOTE: The meta_run method is plugged for the backward pss due to
        # difference in the way arguments are passed to the forward and
        # backward passes to address the memory release issue.
        logging.info("Compiling Backward Graph")
        if self.dist_bwd_gm:
            gm = self.dist_bwd_gm
        bwd_profiler: GraphProfiler = GraphProfiler(
            gm,
            GraphType.BACKWARD,
            fwd_profiler=self.profilers[0][GraphType.FORWARD],
        )
        self.profilers[0][GraphType.BACKWARD] = bwd_profiler

        return make_boxed_func(bwd_profiler.meta_run)

    def run(self, *args, **kwargs) -> None:
        r"""
        Calls the _compile method to initialize the profiler context. Runs
        optional warm-up profiling iterations. This is sometimes essential to
        warm-up the cuda caching allocator and initilize the pinned CPU memory
        when profiling for swapping times as well. Subsequent to warm-up, all
        the profiler statistics are reset and the actual profiling is done for
        number of iterations specified.
        """
        compiled_m = aot_module(
            self.module, self._aot_compile_fwd, self._aot_compile_bwd
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=schedule(
                skip_first=1,
                wait=1,
                warmup=self.warm_up_iters,
                active=self.profile_iters,
            ),
        ) as torch_prof:
            for i in range(2 + self.warm_up_iters + self.profile_iters):
                for profiler in self._iter_all_profilers():
                    profiler.torch_profiler = torch_prof

                if i == 2 + self.warm_up_iters:
                    self.reset_stats()
                self.forward_loss(compiled_m, *args, **kwargs).backward()

                # We have to manually clear the gradient since self.optimizer
                # is optional. See the comment about self.optimizer in CTOR.
                for p in compiled_m.parameters():
                    if p.grad is not None:
                        p.grad = None
                torch_prof.step()

    def _broadcast_rank0_result(self) -> None:
        """
        Broadcast rank0 node_info to all ranks. The current assumption is that
        all ranks will have exactly the same graphs. The design won't work if
        the assumption does not hold. The key is we need to map the name to its
        corresponding fx.Node.

        This API is needed as different profiling may procude different
        bucketing and scheduling. As a result, without the synchronization, the
        forward and backward passes may be stuck due to different ranks hold
        different optimized graphs.
        """
        prof = self.profilers[0][GraphType.FORWARD]
        broadcast_node_info = {}
        node_name_to_nodes = {}

        # Flatten NodeInfo to a tuple that contains hashable values.
        for node, node_info in prof.node_info.items():
            plain_node_info = []
            for field in fields(node_info):
                item = getattr(node_info, field.name)
                if isinstance(item, (torch.Tensor, torch.cuda.Event)):
                    plain_node_info.append(None)
                elif isinstance(item, fx.Node):
                    node_name_to_nodes[item.name] = item
                    plain_node_info.append(item.name)
                else:
                    plain_node_info.append(item)
            node_name_to_nodes[node.name] = node
            broadcast_node_info[node.name] = plain_node_info

        # Broadcast only rank0 result.
        object_list = [broadcast_node_info]
        dist.broadcast_object_list(object_list)
        broadcast_node_info = object_list[0]

        # Unflatten the tuple to a NodeInfo or IntermediateNodeInfo.
        for node in prof.node_info.keys():
            plain_node_info = broadcast_node_info[node.name]
            node_info = prof.node_info[node]
            node_info_fields = fields(node_info)
            node_info_kwargs = {}
            for field, plain_item in zip(node_info_fields, plain_node_info):
                item = getattr(node_info, field.name)
                if isinstance(item, fx.Node):
                    assert isinstance(plain_item, str)
                    node_info_kwargs[field.name] = node_name_to_nodes[
                        plain_item
                    ]
                elif isinstance(item, (torch.Tensor, torch.cuda.Event)):
                    assert item is None
                    node_info_kwargs[field.name] = None
                else:
                    node_info_kwargs[field.name] = item
            cls = (
                IntermediateNodeInfo
                if isinstance(node_info, IntermediateNodeInfo)
                else NodeInfo
            )
            prof.node_info[node] = cls(**node_info_kwargs)

    def summarize(self, to_print: bool = False) -> None:
        r"""
        Aggregates all the statistics accumulated during the profiling
        iterations and makes them ready for printing.
        """
        for profiler in self._iter_all_profilers():
            profiler.summarize()
            if to_print:
                print(profiler.print_summary())
        if self.dist_fwd_gm:
            self._broadcast_rank0_result()

    def reset_stats(self):
        r"""
        Resets all the accumulated profiling statistics. Usualy called after
        warm-up iterations or before beginning a new profiling session.
        """
        for profiler in self._iter_all_profilers():
            profiler.reset_stats()
