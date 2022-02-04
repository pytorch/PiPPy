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

def _get_value_on_remote(rref):
    return rref.local_value()

@rpc.functions.async_execution
def async_transfer(scheduler_rref, rref_arg, arg_idx, runlist_key):
    self = scheduler_rref.local_value()
    fut = rpc.rpc_async(to=rref_arg.owner(), func=_get_value_on_remote, args=(rref_arg,))

    def bottom_half(fut):
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
    * TODO: storage of activations for subsequent gradient computation
    * TODO: gradient checkpointing
    * TODO: Invocation of `torch.autograd.backward()` to implement backward passes
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

    def worker_loop(self):
        while True:
            with self.ready_runlist_cv:
                while len(self.ready_runlist) == 0:
                    self.ready_runlist_cv.wait()

                # TODO: extra priorities
                first_key = next(iter(self.ready_runlist.keys()))
                work_item = self.ready_runlist.pop(first_key)

            work_item.state = SchedState.RUNNING
            args_rrefs = work_item.args
            kwargs_rrefs = work_item.kwargs
            future = work_item.future
            microbatch_id = work_item.microbatch_id
            ready_args = work_item.ready_args
            phase = work_item.phase

            if phase == Phase.BACKWARD:
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
            logging.info(f'[{self.local_rank}][{microbatch_id}] Running WorkItem {work_item.debug_str}')
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
                out_val = self.loss_mod(*args, **kwargs)
                self.loss_cache[microbatch_id] = (out_val, flat_tensor_args)
            elif work_item.phase == Phase.BACKWARD:
                output_grads = kwargs['output_grads']
                out_val = stage_backward(*args, **kwargs)
            else:
                assert False, f'Unrecognized phase {work_item.phase} encountered in execution'

            future.set_result(out_val)
            work_item.state = SchedState.DONE


    @rpc.functions.async_execution
    def invoke(self, phase : Phase, args, kwargs, cur_microbatch : int, debug_str : str):
        # Extract all RRef arguments so we can spawn asynchronous data transfers
        # for each of them
        rref_args : List[torch._C._distributed_rpc.PyRRef] = []
        def extract_rref_args(arg):
            if isinstance(arg, torch._C._distributed_rpc.PyRRef):
                rref_args.append(arg)
        torch.fx.node.map_aggregate(args, extract_rref_args)
        torch.fx.node.map_aggregate(kwargs, extract_rref_args)

        # Construct WorkItem for this microbatch+phase and record it in the
        # waiting runlist
        future = torch.futures.Future()
        # TODO: increase blocked_args_count for extra things like scheduling
        work_item = WorkItem(phase, args, kwargs, future, cur_microbatch, len(rref_args), {}, debug_str=debug_str)
        key_suffix = {
            Phase.FORWARD : 'forward',
            Phase.LOSS : 'loss',
            Phase.BACKWARD : 'backward'
        }
        runlist_key = f'{cur_microbatch}_{key_suffix[phase]}'
        if len(rref_args) == 0:
            # TODO: convert initial input into RRef?
            with self.ready_runlist_cv:
                self.ready_runlist[runlist_key] = work_item
                self.ready_runlist_cv.notify()
        else:
            with self.waiting_runlist_lock:
                assert runlist_key not in self.waiting_runlist, f'key {runlist_key} already in waiting runlist {self.waiting_runlist}'
                self.waiting_runlist[runlist_key] = work_item


        # Spawn asyncronous data transfers for each of the RRef arguments.
        _futures = []
        for arg_idx, rref_arg in enumerate(rref_args):
            self_rref = rpc.RRef(self)
            _futures.append(rpc.rpc_async(
                to=self.local_rank, func=async_transfer, args=(self_rref, rref_arg, arg_idx, runlist_key)))

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
        splits = []
        for i, (arg, batch_dim) in enumerate(zip(args, batch_dims)):
            if isinstance(arg, torch.Tensor):
                if batch_dim is None:
                    raise RuntimeError(f'Batch dimension not specified for arg {i}')

                sizes = self._calc_microbatch_split_sizes(chunks, arg.shape[batch_dim])
                logging.info(f'[root] Splitting arg {i} with size {arg.shape} into chunks {sizes} along dimension {batch_dim}')
                if _debug_mask_minibatches:
                    logging.info(f'[root] Using masked minibatches')
                    chunk_tensors = []

                    prefix_sums = []
                    sum = 0
                    for size in sizes:
                        sum += size
                        prefix_sums.append(sum)

                    predecessor = 0

                    for sum in prefix_sums:
                        splits.append((predecessor, sum))
                        predecessor = sum

                    for start, finish in splits:
                        new_tensor = torch.zeros_like(arg)
                        new_tensor[start:finish] = arg[start:finish]
                        chunk_tensors.append(new_tensor)
                else:
                    logging.info(f'[root] Not using masked minibatches, normal tensor split')
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

        return split_args, splits

def DEBUG_INDEX(arg, idx, debug_str):
    val = arg.local_value()[idx]
    print(f'*****INDEXING VALUE {debug_str} {val}')
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
            return stage_executor.remote().invoke(phase, args, kwargs, self.cur_microbatch, debug_str=node.format_node())
        else:
            logging.info(f'[root][{self.cur_microbatch}] Running local operation {target} from driver')
            return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], torch._C._distributed_rpc.PyRRef):
            return args[0].remote().__getitem__(args[1])
            # return rpc.remote(args[0].owner(), DEBUG_INDEX, (args[0], args[1], self.node_list[self.pc].format_node()))
        elif target is stage_backward:
            node = self.node_list[self.pc]
            assert 'fw_stage' in node.meta
            rank, stage_executor = self.remote_stage_executor_rrefs[node.meta['fw_stage']]
            logging.info(f'[root][{self.cur_microbatch}] Issuing BW invocation for target {node.meta["fw_stage"]} on rank {rank}')
            return stage_executor.remote().invoke(Phase.BACKWARD, args, kwargs, self.cur_microbatch, debug_str=node.format_node())
        return super().call_function(target, args, kwargs)

    def run_until(self, predicate : Callable[[torch.fx.Node], bool]):
        for self.pc in range(self.pc, len(self.node_list)):
            node = self.node_list[self.pc]

            if predicate(node):
                return node

            # TODO: hoist run() implementation
            self.env[node] = super().run_node(node)

class PipelineDriverFillDrain(PipelineDriverBase):
    def __init__(self, pipe : Pipe, world_size : int, all_ranks : List[int] = None, single_loss : bool = False):
        super().__init__(pipe, world_size, all_ranks)
        self.single_loss = single_loss

    def run(self, *args, chunks : int, batch_dims : Optional[List[Optional[int]]] = None,
            _debug_mask_minibatches : bool = False):
        logging.info(f'[root] Running pipeline')
        # Roadmap:
        # 1) Micro-batch splitting - divide input arguments out into concrete chunk values
        # 2) Interpreter tiling - one interpreter per micro-batch
        # 3) Scheduling - Use control logic to advance interpreters to issue round-robin
        #       forward work items, then round robin losses, then round robin backwards

        split_args, splits = self._split_args_into_microbatches(
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
            logging.info(f'[root][i] Executing forward stages')
            last_nodes.append(interp.run_until(node_is_loss_or_output))

        assert all(n == last_nodes[0] for n in last_nodes)

        if last_nodes[0].op == 'output':
            logging.info(f'[root] Program does not have loss/backward, returning outputs directly')
            # Forward-only; return output values
            return self._retrieve_output_values(microbatch_interpreters, last_nodes, _debug_mask_minibatches, splits)
   
        if self.single_loss:
            raise NotImplementedError('Single loss not yet implemented')

        logging.info(f'[root] Executing loss + backward stages')
        last_nodes = []
        for interp in microbatch_interpreters:
            last_nodes.append(interp.run_until(lambda n: n.op == 'output'))

        assert all(n == last_nodes[0] for n in last_nodes)
        assert last_nodes[0].op == 'output'
        return self._retrieve_output_values(microbatch_interpreters, last_nodes, _debug_mask_minibatches, splits)

    def _retrieve_output_values(self, microbatch_interpreters, last_nodes, _debug_mask_minibatches, splits):
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
            logging.info(f'[root] Reducing loss output using `torch.mean`')
            return torch.mean(torch.stack(local_results))

        if _debug_mask_minibatches:
            logging.info(f'[root] Using masked outputs, splicing valid sections')
            assert len(splits) > 0
            sliced_outputs = []
            for result, (start, end) in zip(local_results, splits):
                sliced_outputs.append(result[start:end])
            logging.info(f'[root] Returning spliced outputs')
            return torch.cat(sliced_outputs)

        logging.info(f'Returning concatenated outputs')
        return torch.cat(local_results)


