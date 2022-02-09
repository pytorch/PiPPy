from concurrent.futures import Executor
import torch
import torch.fx
import torch.distributed.rpc as rpc
from IR import Pipe, stage_backward
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, NamedTuple, Optional
import logging
import operator
import threading
import warnings

# Schedule design notes:
#
# At this point we need to impart physical execution semantics onto the abstract IR.
# The IR describes the computation's data and code partitioned into stages, but it
# does not specify micro-batch semantics or the order in which the work is actually
# executed. So at this point we:
#
# 1. Define the strategy for replicating the computation. In particular, we will likely make the assumption
#    that the operations in the program are batch-wise commutative (my term), i.e. we can guarantee equivalence
#    with splitting up the operation along the batch dimension, applying the computation to those sub-batches,
#    then merging them back together via concatenation. We should provide a crisp contract surrounding this
# 2.  Define the way the RemoteInterpreter will interpret the IR and issue `invoke` commands to the
#     `PipeStageExecutor` instances. This will depend entirely on each execution schedule's strategy
#        * FillDrain variants will issue all forward jobs before all backward jobs
#        * 1F1B will issue jobs in some valid topological order
#        * dynamic backpressue is similar to 1f1b
#     * TODO: need to define how information like # pipeline stages, num chunks, etc is communicated to
#           workers
# 3. Define the way that the `PipeStageExecutor` will consume `WorkItems` in its `ready` runlist. This
#    similarly depends entirely on the strategy of the execution schedule.
#        * FillDrain with micro-batch loss will execute all forward WorkItems and come to a "barrier". Once
#          all micro-batches reach this barrier, the scheduler will start executing loss and backward
#          WorkItems
#        * FillDrain with mini-batch loss will do as above^ but internally the scheduler on the last stage
#          will compute the loss by concatenating the results from each micro-batch, applying the loss
#          once (over the whole mini-batch), then executing backward WorkItems with slices of the mini-batch
#          loss.
#        * 1F1B for each stage will compute a fill phase consisting of only forward WorkItems, a steady-state
#          where forward/backward WorkItems are alternated, and a drain phase consisting of backward WorkItems.
#          Last stage will always compute loss and backward after a forward WorkItem.
#        * Dynamic resource-based scheduling will have a set of open "registers" that limit the number of
#          forward WorkItems that can be admitted. A corresponding backward WorkItem will release the
#          registers allocated by the forward.
#
# Idea: compiler/processor split
#       Compiler: orders instructions from each micro-batch into some order for consumption by the processor
#       Processor: configurable execution engine. Mainly can be configured for in-order or out-of-order
#                  execution.
#
#   Fill-drain: Compiler orders all forward chunks, loss, then all backward. Could either be an in-order
#               processor or an out-of-order processor. In the case of OOO, compiler will emit barrier
#               instruction
#   1F1B: Compiler orders chunks in 1f1b order. In-order processor, strict about ordering
#   Dynamic: Compiler orders chunks in any order. Out-of-order processor with registers/resource
#            limits.
    
# ===== Questions to Answer =====
# 1. When does each stage happen?
#       micro-batch splitting: per-invocation or with one fixed chunk size?
#       physical compilation: this depends on micro-batch splitting (for e.g. scheduling
#          so it would have to be ordered after micro-batch splitting
#       runtime: obviously needs to happen at runtime
#
# Conceptually:
#
#   replicated_programs : List[IR] = replicate(chunks)
#   schedule : List[IR] = schedule(replicated_programs)
#   for device_schedule in schedule:
#       for instruction in device_schedule:
#           invoke(rank, instruction)
#
#   `chunks` is the only external dependency that could potentially be used per-invocation.
#   Do we want to:
#       a) Take it as a per-invocation parameter and re-do compilation each time? (-overhead)
#       b) Take it as a one-time initialization parameter and consistently split each
#          batch into a single `chunks` value (-flexibility)
#       c) Allow it to be dynamic but cache compiled policies?
#
#   Decision: We can easily convert (a) to (c), so let's go with (a).
#
#
# LOG: 2/7/2022
#
# Need to reimplement per-chunk splitting to account for a single shared loss invocation. The numerics
# of per-microbatch loss are not numerically equivalent, thus numerical comparisons fail

import signal

_executors = []

def sigint_handler(*args):
    for e in _executors:
        e._debug_print(to_file=True)

signal.signal(signal.SIGINT, sigint_handler)

DEBUG = False

def to_here(a):
    if isinstance(a, torch._C._distributed_rpc.PyRRef):
        return a.to_here()
    else:
        return a

class Phase(Enum):
    FORWARD = 0
    LOSS = 1
    BACKWARD = 2

class SchedState(Enum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    DONE = 3

class WorkItem:
    def __init__(
            self, phase, args, kwargs, future, microbatch_id, blocked_args_count, ready_args, state = SchedState.WAITING, debug_str = ''):
        self.phase, self.args, self.kwargs, self.future, self.microbatch_id, self.blocked_args_count, self.ready_args, self.state, self.debug_str \
            = phase, args, kwargs, future, microbatch_id, blocked_args_count, ready_args, state, debug_str

    phase : Phase
    args : Tuple[Any]
    kwargs : Dict[str, Any]
    future : torch.futures.Future
    microbatch_id : int

    blocked_args_count : int
    ready_args : Dict[int, Any]
    state : SchedState
    debug_str : str

    def __str__(self):
        return f'WorkItem({self.debug_str})'

def _get_value_on_remote(caller_rank, callee_rank, runlist_key, microbatch, rref):
    logging.info(f'[{callee_rank}][{microbatch}] Executing async transfer of value '
                 f'{rref} initiated by rank {caller_rank} for {runlist_key}')
    return rref.local_value()

@rpc.functions.async_execution
def async_transfer(rank, microbatch, scheduler_rref, rref_arg, arg_idx, runlist_key):
    logging.info(f'[{rank}][{microbatch}] vvvvv Starting transfer')
    self = scheduler_rref.local_value()
    fut = rpc.rpc_async(to=rref_arg.owner(), func=_get_value_on_remote,
                        args=(rank, rref_arg.owner().id, runlist_key, microbatch, rref_arg,))

    def bottom_half(fut):
        logging.info(f'[{rank}][{microbatch}] Completing transfer of value {rref_arg} from {rref_arg.owner().id} '
                     f'for runlist item {runlist_key}')
        value = fut.value()
        with self.waiting_runlist_lock:
            work_item = self.waiting_runlist[runlist_key]
            work_item.ready_args[arg_idx] = value
            work_item.blocked_args_count -= 1
            if work_item.blocked_args_count == 0:
                with self.ready_runlist_cv:
                    work_item.state = SchedState.READY
                    self.ready_runlist[runlist_key] = self.waiting_runlist.pop(runlist_key)
                    self.ready_runlist_cv.notify()
                logging.info(f'[{rank}][{microbatch}] vvvvv All operands ready')
            else:
                logging.info(f'[{rank}][{microbatch}] vvvvv Still waiting for {work_item.blocked_args_count} operands.')

    return fut.then(bottom_half)

class PipeStageExecutor:
    """
    PipeStageExecutor encapsulates the execution semantics of a fragement of
    code on a pipeline stage. PipeStageExecutor handles:

    * Ownership of the stage's module and its recursive submodules/parameters
    * Serving as an entrypoint for the driver to push jobs into its queue
    * TODO: in-order execution
    * Queueing of jobs and execution schedule, e.g.
        * Static Schedules
            * TODO: fill-drain pipeline by serializing jobs
            * TODO: 1F1B scheduling by serializing jobs and stalling for a specific
                    phase to come through
            * TODO: Interleaved 1F1B (TODO: how to set up these data dependencies)
        * Dynamic Schedules
            * TODO: Varuna dynamic schedule
            * TODO: dynamic scheduling via registers and back-pressure (TODO: how to
                    specify resource limits and how to implement backpressure?)
    * TODO: gradient checkpointing
    """

    def __init__(self, mod, local_rank, loss_mod=None):
        self.local_rank = local_rank
        logging.info(f'[{self.local_rank}] Instantiating PipeStageExecutor')
        self.mod = mod
        self.loss_mod = loss_mod

        self.waiting_runlist_lock = threading.Lock()
        # self.waiting_rulist (*and the contained WorkItems*) are guarded by
        # self.waiting_runlist_lock
        self.waiting_runlist : Dict[WorkItem, None] = {}

        self.ready_runlist_lock = threading.Lock()
        self.ready_runlist_cv = threading.Condition(self.ready_runlist_lock)
        self.ready_runlist : Dict[str, WorkItem] = {}

        self.worker_thread = threading.Thread(target=self.worker_loop, name=f'worker_{self.mod}', daemon=True)
        self.worker_thread.start()

        # map microbatch ID to list of forward tensor args
        self.fwd_cache : Dict[int, Tuple[Any, List[torch.Tensor]]]= {}
        self.loss_cache : Dict[int, Tuple[Any, List[torch.Tensor]]]= {}

        _executors.append(self)

    def _debug_print(self, to_file=False):
        # NB: this does not take the runlist locks. This method should only be
        # called when the system is stalled
        s = f'Executor instance {id(self)} for rank {self.local_rank}.\n' \
            f'\tWaiting WorkItems: {self.waiting_runlist.keys()}\n' \
            f'\tReady WorkItems: {self.ready_runlist.keys()}\n'

        blocked_args = {}
        ready_args = {}
        for name, workitem in self.waiting_runlist.items():
            if workitem.blocked_args_count > 0:
                pass
                rref_args = []
                def retrieve_rref_args_by_idx(a):
                    if isinstance(a, torch._C._distributed_rpc.PyRRef):
                        rref_args.append(a)
                torch.fx.node.map_aggregate(workitem.args, retrieve_rref_args_by_idx)
                torch.fx.node.map_aggregate(workitem.kwargs, retrieve_rref_args_by_idx)
                blocked_rref_idxs = set(range(len(rref_args))) - set(workitem.ready_args.keys())
                blocked_args[name] = blocked_rref_idxs
                ready_args[name] = workitem.ready_args.keys()

        s += f'\tBlocked args: {blocked_args}\n'
        s += f'\tReady args: {ready_args}\n'

        if to_file:
            with open(f'{self.local_rank}.log', 'w') as f:
                f.write(s)

        return s

    def worker_loop(self):
        while True:
            with self.ready_runlist_cv:
                while len(self.ready_runlist) == 0:
                    self.ready_runlist_cv.wait()

                logging.info(f'[{self.local_rank}] Dequeueing workitem from set of {len(self.ready_runlist)}')
                # TODO: extra priorities
                first_key = next(iter(self.ready_runlist.keys()))
                work_item = self.ready_runlist.pop(first_key)

            logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Got WorkItem {work_item}')

            work_item.state = SchedState.RUNNING
            args_rrefs = work_item.args
            kwargs_rrefs = work_item.kwargs
            future = work_item.future
            microbatch_id = work_item.microbatch_id
            ready_args = work_item.ready_args
            phase = work_item.phase

            if phase == Phase.BACKWARD:
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running backward phase. Retrieving stashed values')
                # HACK: here we are directly accessing the saved tensor outputs
                # for closed-over outputs so that they still have the grad_fn
                # from local autograd. Can we solve this more elegantly?
                cache = self.loss_cache if len(self.loss_cache) > 0 else self.fwd_cache
                kwargs_rrefs = dict(kwargs_rrefs)
                kwargs_rrefs['stage_output'], kwargs_rrefs['input_values'] = cache.pop(microbatch_id)

            rref_arg_idx = 0
            def retrieve_rref_args_by_idx(a):
                if isinstance(a, torch._C._distributed_rpc.PyRRef):
                    nonlocal rref_arg_idx
                    val = ready_args[rref_arg_idx]
                    rref_arg_idx += 1
                    return val
                else:
                    return a

            args = torch.fx.node.map_aggregate(args_rrefs, retrieve_rref_args_by_idx)
            kwargs = torch.fx.node.map_aggregate(kwargs_rrefs, retrieve_rref_args_by_idx)
            if work_item.phase == Phase.FORWARD:
                flat_tensor_args = []
                def extract_tensor_args(a):
                    if isinstance(a, torch.Tensor):
                        nonlocal flat_tensor_args
                        val = a.detach().requires_grad_(a.requires_grad)
                        flat_tensor_args.append(val)
                        return val
                    else:
                        return a
                args = torch.fx.node.map_aggregate(args, extract_tensor_args)
                kwargs = torch.fx.node.map_aggregate(kwargs, extract_tensor_args)
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running forward module')
                out_val = self.mod(*args, **kwargs)
                self.fwd_cache[microbatch_id] = (out_val, flat_tensor_args)
            elif work_item.phase == Phase.LOSS:
                assert self.loss_mod is not None
                flat_tensor_args = []
                def extract_tensor_args(a):
                    if isinstance(a, torch.Tensor):
                        nonlocal flat_tensor_args
                        val = a.detach().requires_grad_(a.requires_grad)
                        flat_tensor_args.append(val)
                        return val
                    else:
                        return a
                args = torch.fx.node.map_aggregate(args, extract_tensor_args)
                kwargs = torch.fx.node.map_aggregate(kwargs, extract_tensor_args)
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running loss module')
                out_val = self.loss_mod(*args, **kwargs)
                self.loss_cache[microbatch_id] = (out_val, flat_tensor_args)
            elif work_item.phase == Phase.BACKWARD:
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running backward')
                out_val = stage_backward(*args, **kwargs)
            else:
                assert False, f'Unrecognized phase {work_item.phase} encountered in execution'

            logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Populating result of type {type(out_val)} '
                         f'for {first_key}')
            future.set_result(out_val)
            work_item.state = SchedState.DONE

    @rpc.functions.async_execution
    def invoke(self, unique_key : str, phase : Phase, args, kwargs, cur_microbatch : int, debug_str : str):
        # TODO: do we need to serialize calls to invoke() to preserve the order in which WorkItems appear for
        # static schedules?

        logging.info(f'[{self.local_rank}][{cur_microbatch}] Received invoke call for {debug_str}')
        # Extract all RRef arguments so we can spawn asynchronous data transfers
        # for each of them
        rref_args : List[torch._C._distributed_rpc.PyRRef] = []
        def extract_rref_args(arg):
            if isinstance(arg, torch._C._distributed_rpc.PyRRef):
                rref_args.append(arg)
        torch.fx.node.map_aggregate(args, extract_rref_args)
        torch.fx.node.map_aggregate(kwargs, extract_rref_args)

        logging.info(f'[{self.local_rank}][{cur_microbatch}] Invoke call found {len(rref_args)} RRef arguments')

        # Construct WorkItem for this microbatch+phase and record it in the
        # waiting runlist
        future = torch.futures.Future()
        # TODO: increase blocked_args_count for extra things like scheduling
        work_item = WorkItem(phase, args, kwargs, future, cur_microbatch, len(rref_args), {}, debug_str=debug_str)
        logging.info(f'[{self.local_rank}][{cur_microbatch}] Invoke instantiated WorkItem {work_item} with key {unique_key}')
        if len(rref_args) == 0:
            # TODO: convert initial input into RRef?
            logging.info(f'[{self.local_rank}][{cur_microbatch}] No RRef arguments. Scheduling directly as READY workitem')
            work_item.state = SchedState.READY
            with self.ready_runlist_cv:
                logging.info(f'[{self.local_rank}][{cur_microbatch}] Current ready runlist keys: {self.ready_runlist.keys()}')
                self.ready_runlist[unique_key] = work_item
                self.ready_runlist_cv.notify()
        else:
            logging.info(f'[{self.local_rank}][{cur_microbatch}] Scheduling WorkItem as WAITING workitem')
            work_item.state = SchedState.WAITING
            with self.waiting_runlist_lock:
                logging.info(f'[{self.local_rank}][{cur_microbatch}] Current waiting runlist keys: {self.waiting_runlist.keys()}')
                assert unique_key not in self.waiting_runlist, f'key {unique_key} already in waiting runlist {self.waiting_runlist}'
                self.waiting_runlist[unique_key] = work_item


        # Spawn asyncronous data transfers for each of the RRef arguments.
        _futures = []
        for arg_idx, rref_arg in enumerate(rref_args):
            logging.info(f'[{self.local_rank}][{cur_microbatch}] Launching asynchronous data transfer for RRef {arg_idx} {rref_arg}')
            self_rref = rpc.RRef(self)
            _futures.append(rpc.rpc_async(
                to=self.local_rank, func=async_transfer,
                args=(self.local_rank, cur_microbatch, self_rref, rref_arg, arg_idx, unique_key)))

        if DEBUG:
            # Make exceptions visible
            for fut in _futures:
                fut.wait()

        return future


class PipelineDriverBase:
    def __init__(self, pipe : Pipe, world_size : int, all_ranks : List[int] = None):
        self.pipe = pipe
        self.world_size = world_size
        self.all_ranks = all_ranks
        self.executor_class = PipeStageExecutor

        self._init_remote_executors()

    def _init_remote_executors(self):
        self.remote_stage_executor_rrefs : Dict[str, torch.distributed.rpc.RRef] = {}

        if self.all_ranks is not None:
            assert len(self.all_ranks) == self.world_size, "Explicitly specified ranks must match world_size"
        else:
            self.all_ranks = list(range(self.world_size))

        class ExecutorDescriptor:
            name : str
            mod : torch.nn.Module
            loss_mod : Optional[torch.nn.Module] = None
            has_backward : bool = False

        split_gm = self.pipe.split_gm

        executor_descriptors = []
        seen_loss = False
        seen_loss_backward = False
        bw_idx = -1
        for node in split_gm.graph.nodes:
            if node.op == 'call_module':
                assert not seen_loss
                target_mod = split_gm.get_submodule(node.target)
                if node.target == '_loss':
                    assert len(executor_descriptors) > 0
                    executor_descriptors[-1].loss_mod = target_mod
                else:
                    descr = ExecutorDescriptor()
                    descr.name = node.target
                    descr.mod = target_mod
                    executor_descriptors.append(descr)
            elif (node.op, node.target) == ('call_function', stage_backward):
                if not seen_loss_backward:
                    seen_loss_backward = True
                    node.meta['fw_stage'] = '_loss'
                else:
                    executor_descriptors[bw_idx].has_backward = True
                    node.meta['fw_stage'] = executor_descriptors[bw_idx].name
                    bw_idx -= 1

        assert all(d.has_backward for d in executor_descriptors) or all(not d.has_backward for d in executor_descriptors)

        if len(executor_descriptors) > self.world_size:
            raise RuntimeError(f'Tried to run pipeline with {len(executor_descriptors)} stages with a world size of '
                               f'{self.world_size}. Please ensure world_size is large enough to accomodate your pipeline.')

        if len(executor_descriptors) < self.world_size:
            warnings.warn(f'Running pipeline with {len(executor_descriptors)} stages on world_size of {self.world_size}. '
                          f'Remaining ranks will be idle.')

        self.loss_stage = None
        for rank, descr in zip(self.all_ranks, executor_descriptors):
            self.remote_stage_executor_rrefs[descr.name] = (rank, rpc.remote(rank, self.executor_class, (descr.mod, rank, descr.loss_mod)))
            if descr.loss_mod:
                self.remote_stage_executor_rrefs['_loss'] = self.remote_stage_executor_rrefs[descr.name]
        
    def run(self, *args, chunks : int, batch_dims : Optional[List[Optional[int]]] = None,
            _debug_mask_minibatches : bool = False):
        raise NotImplementedError('PipelineDriverBase is an abstract base class, please use a concrete '
                                  'implementation class.')

    class MicroBatchSplitTensor(NamedTuple):
        chunks : List[torch.Tensor]

    def _calc_microbatch_split_sizes(self, chunks : int, dim_size : int):
        # TODO: this splits with the last one bigger because i can't
        # figure out the math to make the last one smaller
        chunk_size = dim_size // chunks

        sizes = []
        examples_counted = 0
        for i in range(chunks):
            if i == chunks - 1:
                sizes.append(dim_size - examples_counted)
                examples_counted += (dim_size - examples_counted)
            else:
                sizes.append(chunk_size)
                examples_counted += chunk_size

        assert examples_counted == dim_size
        return sizes

    def _split_args_into_microbatches(self, *args, chunks : int, batch_dims : Optional[List[Optional[int]]] = None,
                                      _debug_mask_minibatches : bool = False):
        logging.info(f'[root] Splitting args with sizes '
                     f'{[arg.shape if isinstance(arg, torch.Tensor) else arg for arg in args]} into {chunks} chunks.')
        # Calculate full batch dims array
        if batch_dims is None:
            batch_dims = [0 if isinstance(arg, torch.Tensor) else None for arg in args]
        assert isinstance(batch_dims, list)
        logging.info(f'[root] Arguments have batch dims {batch_dims}')

        if len(args) != len(batch_dims):
            raise RuntimeError('Length of `batch_dims` must match')
        split_args = []

        # TODO: assuming input splits are the same as outputs
        splits_per_arg = []
        for i, (arg, batch_dim) in enumerate(zip(args, batch_dims)):
            if isinstance(arg, torch.Tensor):
                if batch_dim is None:
                    raise RuntimeError(f'Batch dimension not specified for arg {i}')

                sizes = self._calc_microbatch_split_sizes(chunks, arg.shape[batch_dim])
                logging.info(f'[root] Splitting arg {i} with size {arg.shape} into chunks {sizes} along dimension {batch_dim}')
                if _debug_mask_minibatches:
                    splits = []
                    chunk_tensors = []

                    prefix_sums = []
                    sum = 0
                    for size in sizes:
                        sum += size
                        prefix_sums.append(sum)

                    logging.info(f'[root] Prefix sums {prefix_sums}')

                    predecessor = 0

                    for sum in prefix_sums:
                        splits.append((predecessor, sum))
                        predecessor = sum

                    logging.info(f'[root] splits {splits}')

                    for start, finish in splits:
                        new_tensor = torch.zeros_like(arg)
                        new_tensor[start:finish] = arg[start:finish]
                        chunk_tensors.append(new_tensor)
                    logging.info(f'[root] Chunk tensor sizes {[t.shape for t in chunk_tensors]}')

                    splits_per_arg.append(splits)
                else:
                    chunk_tensors = torch.split(arg, sizes)

                split_args.append(self.MicroBatchSplitTensor(chunk_tensors))

            else:
                logging.info(f'[root] Arg {i} is a non-tensor value, not splitting')
                split_args.append(arg)

        def split_str(a):
            if isinstance(a, self.MicroBatchSplitTensor):
                return f'MicrobatchSplitTensor(chunks={[c.shape for c in a.chunks]}'
            else:
                return str(a)
        logging.info(f'[root] Final splits: {[split_str(a) for a in split_args]}')

        return split_args, splits_per_arg

def DEBUG_INDEX(rank, microbatch, arg, idx, debug_str):
    indexee = arg.local_value()
    val = indexee[idx]
    logging.info(f'[{rank}][{microbatch}] *****INDEXING VALUE {debug_str} input_type {type(indexee)} index {idx} output type {type(val)}')
    return val

class RemoteInterpreter(torch.fx.Interpreter):
    def __init__(self, remote_stage_executor_rrefs, module, cur_microbatch : int, init_args, garbage_collect_values = True):
        super().__init__(module, garbage_collect_values)
        self.remote_stage_executor_rrefs = remote_stage_executor_rrefs
        self.cur_microbatch = cur_microbatch
        self.args_iter = iter(init_args)
        self.pc = 0
        self.node_list = list(self.module.graph.nodes)

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        node = self.node_list[self.pc]

        if target in self.remote_stage_executor_rrefs:
            rank, stage_executor = self.remote_stage_executor_rrefs[target]
            phase = Phase.LOSS if target == '_loss' else Phase.FORWARD
            logging.info(f'[root][{self.cur_microbatch}] Issuing {phase} invocation for target {target} on rank {rank}')
            invocation_key = f'{self.cur_microbatch}_{node.name}'
            return stage_executor.remote().invoke(invocation_key, phase, args, kwargs, self.cur_microbatch, debug_str=node.format_node())
        else:
            logging.info(f'[root][{self.cur_microbatch}] Running local operation {target} from driver')
            return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], torch._C._distributed_rpc.PyRRef):
            # return args[0].remote().__getitem__(args[1])
            # HACK: wtf is going on here
            return rpc.rpc_sync(args[0].owner(), DEBUG_INDEX, (args[0].owner().id, self.cur_microbatch, args[0], args[1], self.node_list[self.pc].format_node()))
        elif target is stage_backward:
            node = self.node_list[self.pc]
            assert 'fw_stage' in node.meta
            rank, stage_executor = self.remote_stage_executor_rrefs[node.meta['fw_stage']]
            logging.info(f'[root][{self.cur_microbatch}] Issuing BW invocation for target {node.meta["fw_stage"]} on rank {rank}')
            invocation_key = f'{self.cur_microbatch}_{node.name}'
            return stage_executor.remote().invoke(invocation_key, Phase.BACKWARD, args, kwargs, self.cur_microbatch, debug_str=node.format_node())
        return super().call_function(target, args, kwargs)

    def run_until(self, predicate : Callable[[torch.fx.Node], bool]):
        for self.pc in range(self.pc, len(self.node_list)):
            node = self.node_list[self.pc]

            if predicate(node):
                return node

            # TODO: hoist run() implementation
            logging.info(f'[{self.cur_microbatch}] Issue command to run {node.format_node()}')
            self.env[node] = super().run_node(node)

class PipelineDriverFillDrain(PipelineDriverBase):
    def __init__(self, pipe : Pipe, world_size : int, all_ranks : List[int] = None, single_loss : bool = False):
        super().__init__(pipe, world_size, all_ranks)
        self.single_loss = single_loss

    def run(self, *args, chunks : int, batch_dims : Optional[List[Optional[int]]] = None,
            _debug_mask_minibatches : bool = False):
        if self.single_loss:
            raise NotImplementedError('Single minibatch loss not implemented')

        logging.info(f'[root] Running pipeline')
        # Roadmap:
        # 1) Micro-batch splitting - divide input arguments out into concrete chunk values
        # 2) Interpreter tiling - one interpreter per micro-batch
        # 3) Scheduling - Use control logic to advance interpreters to issue round-robin
        #       forward work items, then round robin losses, then round robin backwards

        split_args, splits_per_arg = self._split_args_into_microbatches(
            *args, chunks=chunks, batch_dims=batch_dims, _debug_mask_minibatches=_debug_mask_minibatches)

        microbatch_interpreters : List[self.RunUntilInterpreter] = []

        for chunk in range(chunks):
            logging.info(f'[root] Instantiating microbatch interpreter for chunk {chunk}') 
            initial_arg_chunks = [arg.chunks[chunk] for arg in split_args]
            interp = RemoteInterpreter(self.remote_stage_executor_rrefs, self.pipe.split_gm, chunk, initial_arg_chunks)
            microbatch_interpreters.append(interp)

        logging.info(f'[root] {len(microbatch_interpreters)} instantiated')

        def node_is_loss_or_output(n):
            return (n.op == 'call_module' and n.target == '_loss') or n.op == 'output'

        last_nodes = []
        for i, interp in enumerate(microbatch_interpreters):
            logging.info(f'[root][{i}] Executing forward stages')
            last_nodes.append(interp.run_until(node_is_loss_or_output))

        assert all(n == last_nodes[0] for n in last_nodes)

        if last_nodes[0].op == 'output':
            logging.info(f'[root] Program does not have loss/backward, returning outputs directly')
            # Forward-only; return output values
            return self._retrieve_output_values(microbatch_interpreters, last_nodes, _debug_mask_minibatches, splits_per_arg)
   
        logging.info(f'[root] Executing loss + backward stages')
        last_nodes = []
        for interp in microbatch_interpreters:
            last_nodes.append(interp.run_until(lambda n: n.op == 'output'))

        assert all(n == last_nodes[0] for n in last_nodes)
        assert last_nodes[0].op == 'output'
        return self._retrieve_output_values(microbatch_interpreters, last_nodes, _debug_mask_minibatches, splits_per_arg)

    def _retrieve_output_values(self, microbatch_interpreters, last_nodes, _debug_mask_minibatches, splits_per_arg):
        logging.info(f'[root] Combining output values from {len(microbatch_interpreters)} chunks')
        output_vals = []
        for interp, last_node in zip(microbatch_interpreters, last_nodes):
            interp.run_until(lambda n : False)
            output_vals.append(interp.env[last_node])

        # TODO: non-single-output returns?
        local_results = [to_here(result) for result in output_vals]
        logging.info(f'[root] Got {len(local_results)} outputs')

        if all(isinstance(r, torch.Tensor) and r.ndim == 0 for r in local_results):
            # HACK - design more systematic programming model for losses, which
            # reduce
            return torch.sum(torch.stack(local_results))

        if _debug_mask_minibatches:
            logging.info(f'[root] Using masked outputs, splicing valid sections')
            assert len(splits_per_arg) > 0
            # HACK: assuming split is the same as split for first arg
            splits = splits_per_arg[0]
            sliced_outputs = []
            for result, (start, end) in zip(local_results, splits):
                sliced_outputs.append(result[start:end])
            logging.info(f'[root] Returning spliced outputs')
            return torch.cat(sliced_outputs)

        logging.info(f'Returning concatenated outputs')
        return torch.cat(local_results)


