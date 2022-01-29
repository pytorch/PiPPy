import torch
import torch.fx
import torch.distributed.rpc as rpc
from enum import Enum
import logging
import operator
import threading
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from IR import Pipe

DEBUG = False

def to_here(a):
    if isinstance(a, torch._C._distributed_rpc.PyRRef):
        return a.to_here()
    else:
        return a

class SchedState(Enum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    DONE = 3

class WorkItem:
    def __init__(
            self, args, kwargs, future, microbatch_id, blocked_args_count, ready_args, state = SchedState.WAITING):
        self.args, self.kwargs, self.future, self.microbatch_id, self.blocked_args_count, self.ready_args, self.state \
            = args, kwargs, future, microbatch_id, blocked_args_count, ready_args, state

    args : Tuple[Any]
    kwargs : Dict[str, Any]
    future : torch.futures.Future
    microbatch_id : int

    blocked_args_count : int
    ready_args : Dict[int, Any]
    state : SchedState

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

    def __init__(self, mod, local_rank):
        self.local_rank = local_rank
        logging.info(f'Rank {self.local_rank} Instantiating PipeStageExecutor for module {mod}')
        self.mod = mod

        self.waiting_runlist_lock = threading.Lock()
        # self.waiting_rulist (*and the contained WorkItems*) are guarded by
        # self.waiting_runlist_lock
        self.waiting_runlist : Dict[WorkItem, None] = {}

        self.ready_runlist_lock = threading.Lock()
        self.ready_runlist_cv = threading.Condition(self.ready_runlist_lock)
        self.ready_runlist : Dict[str, WorkItem] = {}

        self.worker_thread = threading.Thread(target=self.worker_loop, name=f'worker_{self.mod}', daemon=True)
        self.worker_thread.start()

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
            logging.info(f'rank {self.local_rank} running microbatch {microbatch_id} target {self.mod}')
            out_val = self.mod(*args, **kwargs)

            future.set_result(out_val)
            work_item.state = SchedState.DONE

    @rpc.functions.async_execution
    def invoke(self, args, kwargs, cur_microbatch : int):
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
        work_item = WorkItem(args, kwargs, future, cur_microbatch, len(rref_args), {})
        runlist_key = f'{cur_microbatch}_forward'
        if len(rref_args) == 0:
            # TODO: convert initial input into RRef?
            with self.ready_runlist_cv:
                self.ready_runlist[runlist_key] = work_item
                self.ready_runlist_cv.notify()
        else:
            with self.waiting_runlist_lock:
                # TODO: backward jobs
                assert runlist_key not in self.waiting_runlist
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

class RemoteInterpreter(torch.fx.Interpreter):
    def __init__(self, remote_stage_executor_rrefs, module, garbage_collect_values = True):
        super().__init__(module, garbage_collect_values)
        self.remote_stage_executor_rrefs = remote_stage_executor_rrefs
        self.cur_microbatch = -1

    class MicroBatchSplitTensor(NamedTuple):
        chunks : List[torch.Tensor]

    def run(self, *args, chunks : int, batch_dims : Optional[List[Optional[int]]] = None,
            initial_env : Optional[Dict[torch.fx.Node, Any]] = None,
            _debug_mask_minibatches : bool = False):
        """
        TODO: fill this out better

        chunks : number of chunks
        batch_dims : dimension indices for batch dimension for each tensor argument. If None,
                        specified, defaults to dimension `0` for all Tensor arguments. If specified,
                        values can be None to specify non-tensor arguments.
        """

        # Calculate full batch dims array
        if batch_dims is None:
            batch_dims = [0 if isinstance(arg, torch.Tensor) else None for arg in args]
        assert isinstance(batch_dims, list)

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
                if _debug_mask_minibatches:
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
                    chunk_tensors = torch.split(arg, sizes)

                split_args.append(self.MicroBatchSplitTensor(chunk_tensors))

                logging.info(f'Split tensor argument {i} into {chunks} chunks of sizes '
                                f'{[chunk.shape for chunk in chunk_tensors]}')
            else:
                split_args.append(arg)

        microbatch_results = []
        for self.cur_microbatch in range(chunks):
            microbatch_args = []
            for arg in split_args:
                microbatch_args.append(arg.chunks[self.cur_microbatch] if isinstance(arg, self.MicroBatchSplitTensor) else arg)
            microbatch_results.append(super().run(*microbatch_args, initial_env))

        # TODO: figure out what to do here for loss + backward. 
        # TODO: support multiple outputs
        assert all(isinstance(result, torch._C._distributed_rpc.PyRRef) for result in microbatch_results)

        # TODO: make this less hacky. specify output batch_dims?
        local_results = [to_here(result) for result in microbatch_results]

        if _debug_mask_minibatches:
            assert len(splits) > 0
            sliced_outputs = []
            for result, (start, end) in zip(local_results, splits):
                sliced_outputs.append(result[start:end])
            return torch.cat(sliced_outputs)

        return torch.cat(local_results)

    def run_node(self, n):
        with torch.autograd.profiler.record_function(f'run_node {n}'):
            return super().run_node(n)

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

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)

        if target in self.remote_stage_executor_rrefs:
            rank, stage_executor = self.remote_stage_executor_rrefs[target]
            logging.info(f'Issuing remote invocation for target {target} on  rank {rank}')
            return stage_executor.remote().invoke(args, kwargs, self.cur_microbatch)
        else:
            logging.info(f'Running local operation {target} from driver')
            return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], torch._C._distributed_rpc.PyRRef):
            return args[0].remote().__getitem__(args[1])
        return super().call_function(target, args, kwargs)

class PipelineSchedule(Enum):
    FillDrainMiniBatchLoss = 1
    FillDrainMicroBatchLoss = 2
    _1F1BDrain = 3
    DynamicResources = 4

class PipelineDriver:
    def __init__(self, pipe : Pipe, world_size : int, all_ranks : List[int] = None,
                 schedule : PipelineSchedule = PipelineSchedule.FillDrainMiniBatchLoss):
        self.pipe = pipe
        self.world_size = world_size

        if schedule != PipelineSchedule.FillDrainMiniBatchLoss:
            raise NotImplementedError(f'Schedule {schedule} not currently supported!')

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

        self.remote_stage_executor_rrefs : Dict[str, torch.distributed.rpc.RRef] = {}

        if all_ranks is not None:
            assert len(all_ranks) == world_size, "Explicitly specified ranks must match world_size"
        else:
            all_ranks = list(range(world_size))

        # TODO: generalize mapping from pipeline stage to rank
        pipeline_submods = {name : mod for name, mod in self.pipe.split_gm.named_children() if name.startswith('submod_') }

        if len(pipeline_submods) > world_size:
            raise RuntimeError(f'Tried to run pipeline with {len(pipeline_submods)} stages with a world size of '
                               f'{world_size}. Please ensure world_size is large enough to accomodate your pipeline.')

        if len(pipeline_submods) < world_size:
            warnings.warn(f'Running pipeline with {len(pipeline_submods)} stages on world_size of {world_size}. '
                          f'Remaining ranks will be idle.')


        for rank, stage_name in zip(all_ranks, pipeline_submods):
            stage_submod = pipeline_submods[stage_name]
            self.remote_stage_executor_rrefs[stage_name] = (rank, rpc.remote(rank, PipeStageExecutor, (stage_submod, rank)))

        self.interp = RemoteInterpreter(self.remote_stage_executor_rrefs, self.pipe.split_gm)


    def run(self, *args, chunks : int, batch_dims : Optional[List[Optional[int]]] = None,
            initial_env : Optional[Dict[torch.fx.Node, Any]] = None,
            _debug_mask_minibatches : bool = False):
        return self.interp.run(*args, chunks=chunks, batch_dims=batch_dims, initial_env=initial_env,
                              _debug_mask_minibatches=_debug_mask_minibatches)
