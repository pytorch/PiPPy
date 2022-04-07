# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
import random
import threading
import time
import warnings
from enum import Enum
from inspect import Parameter, Signature
from typing import Any, Callable, Dict, List, Tuple, Optional

import torch
import torch.distributed.rpc as rpc
import torch.fx

from pippy.IR import Pipe, stage_backward, sync_barrier
from pippy.microbatch import split_args_kwargs_into_chunks, merge_chunks

# TODO: Define the strategy for replicating the computation. In particular, we will likely make the assumption
# that the operations in the program are batch-wise commutative (my term), i.e. we can guarantee equivalence
# with splitting up the operation along the batch dimension, applying the computation to those sub-batches,
# then merging them back together via concatenation. We should provide a crisp contract surrounding this

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
    BACKWARD = 1
    SYNC_BARRIER = 2

# TODO: do we need this?
class SchedState(Enum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    DONE = 3

class WorkItem:
    def __init__(
            self, phase, args, kwargs, future, microbatch_id, blocked_args_count, ready_args,
            batch_id, num_microbatches, state=SchedState.WAITING, debug_str=''):
        args_to_fwd = ['phase', 'args', 'kwargs', 'future', 'microbatch_id', 'blocked_args_count',
                       'ready_args', 'batch_id', 'num_microbatches', 'state', 'debug_str']

        for arg in args_to_fwd:
            setattr(self, arg, locals()[arg])

    phase : Phase
    args : Tuple[Any]
    kwargs : Dict[str, Any]
    future : torch.futures.Future
    microbatch_id : int

    blocked_args_count : int
    ready_args : Dict[int, Any]
    state : SchedState
    debug_str : str

    batch_id : int
    num_microbatches : int

    def __str__(self):
        return f'WorkItem({self.debug_str})'

    def set_trigger_state(self, max_outstanding):
        if self.phase == Phase.FORWARD and max_outstanding is not None:
            # The pipe schedule has a max outstanding limitation, we will let
            # worker_loop decide whether this forward item can start to run
            self.state = SchedState.WAITING
        else:
            self.state = SchedState.READY


class Event:
    def __init__(self,
                 rank: int,
                 start_ts: float,
                 finish_ts: float,
                 id: Optional[str] = None,
                 name: Optional[str] = None,
                 type: Optional[Any] = None,
                 mbid: Optional[Any] = None
                 ):
        args_to_fwd = ['rank', 'start_ts', 'finish_ts', 'id', 'name', 'type', 'mbid']

        for arg in args_to_fwd:
            setattr(self, arg, locals()[arg])

    rank: int
    start_ts: float
    finish_ts: float
    id: Optional[str]
    name: Optional[str]
    type: Optional[Any]
    mbid: Optional[Any]


def _get_value_on_remote(caller_rank, callee_rank, runlist_key, microbatch, rref):
    logging.info(f'[{callee_rank}][{microbatch}] Executing async transfer of value '
                 f'{rref} initiated by rank {caller_rank} for {runlist_key}')
    return rref.local_value()

@rpc.functions.async_execution
def async_transfer(rank, microbatch, scheduler_rref, rref_arg, arg_idx, runlist_key, max_outstanding):
    logging.info(f'[{rank}][{microbatch}] Starting transfer')
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
                    work_item.set_trigger_state(max_outstanding)
                    self.ready_runlist[runlist_key] = self.waiting_runlist.pop(runlist_key)
                    self.ready_runlist_cv.notify()
                state_str = 'WAITING' if work_item.state == SchedState.WAITING else 'READY'
                logging.info(f'[{rank}][{microbatch}] All operands ready, initialize as {state_str} workitem')
            else:
                logging.info(f'[{rank}][{microbatch}] Still waiting for {work_item.blocked_args_count} operands.')

    return fut.then(bottom_half)

class PipeStageExecutor:
    """
    PipeStageExecutor encapsulates the execution semantics of a fragment of
    code on a pipeline stage. PipeStageExecutor handles:

    * Ownership of the stage's module and its recursive submodules/parameters
    * Serving as an entrypoint for the driver to push jobs into its queue
    * TODO: in-order execution
    * Queueing of jobs and execution schedule, e.g.
        * Static Schedules
            * Fill-drain (GPipe) pipeline by serializing jobs
            * TODO: 1F1B scheduling by serializing jobs and stalling for a specific
                    phase to come through
            * TODO: Interleaved 1F1B (TODO: how to set up these data dependencies)
        * Dynamic Schedules
            * TODO: Varuna dynamic schedule
            * TODO: dynamic scheduling via registers and back-pressure (TODO: how to
                    specify resource limits and how to implement backpressure?)
    * TODO: gradient checkpointing
    """

    def __init__(self, mod, local_rank, max_outstanding=None, dp_pg_cb=None, pp_rank=None):
        self.local_rank = local_rank
        logging.info(f'[{self.local_rank}] Instantiating PipeStageExecutor')
        self.dp_pg_cb = dp_pg_cb

        if self.dp_pg_cb is not None:
            assert pp_rank is not None
            self.mod = torch.nn.parallel.DistributedDataParallel(mod, process_group=self.dp_pg_cb(pp_rank))
        else:
            self.mod = mod

        # Maximum outstanding micro-batches of the pipeline schedule
        self.max_outstanding = max_outstanding
        # Keeps track of the outstanding micro-batches in current executor
        self.outstanding = 0

        self.waiting_runlist_lock = threading.Lock()
        # self.waiting_runlist (*and the contained WorkItems*) are guarded by
        # self.waiting_runlist_lock
        self.waiting_runlist : Dict[str, WorkItem] = {}

        self.ready_runlist_lock = threading.Lock()
        self.ready_runlist_cv = threading.Condition(self.ready_runlist_lock)
        self.ready_runlist : Dict[str, WorkItem] = {}

        self.worker_thread = threading.Thread(target=self.worker_loop, name=f'worker_{self.mod}', daemon=True)
        self.worker_thread.start()

        # map microbatch ID to list of forward tensor args
        self.fwd_cache : Dict[int, Tuple[Any, List[torch.Tensor]]] = {}

        self.events: List[Event] = []

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
        batch_id_to_remaining_backwards_stages : Dict[int, int] = {}

        while True:
            work_item = None
            with self.ready_runlist_cv:
                while len(self.ready_runlist) == 0:
                    self.ready_runlist_cv.wait()

                logging.info(f'[{self.local_rank}] Dequeueing workitem from set of {len(self.ready_runlist)}')
                # TODO: extra priorities
                for key in iter(self.ready_runlist.keys()):
                    # Skip forward work items if we hit the max outstanding limit
                    # If there are no other READY WorkItems, the runloop wraps around to the beginning and blocks again,
                    # waiting for another scheduled WorkItem to wake it back up. This works because the only condition
                    # that can schedule a WAITING Workitem is if another backward WorkItem executes and reduces the number
                    # of outstanding mciro-batches;
                    # If there are other READY WorkItems, the runloop executes as normally processing those
                    if (self.max_outstanding is not None and
                            self.ready_runlist[key].state == SchedState.WAITING and
                            self.outstanding >= self.max_outstanding):
                        continue
                    work_item = self.ready_runlist.pop(key)
                    break

            # We may not fetch any actionable work item in the above loop, go
            # back to the loop in this case
            if work_item is None:
                continue
            logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Got WorkItem {work_item}')

            work_item.state = SchedState.RUNNING
            args_rrefs = work_item.args
            kwargs_rrefs = work_item.kwargs
            future = work_item.future
            microbatch_id = work_item.microbatch_id
            ready_args = work_item.ready_args
            phase = work_item.phase

            batch_id = work_item.batch_id
            num_microbatches = work_item.num_microbatches

            if batch_id not in batch_id_to_remaining_backwards_stages:
                batch_id_to_remaining_backwards_stages[batch_id] = num_microbatches

            start_ts = time.time()

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

            if phase == Phase.BACKWARD:
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running backward phase. Retrieving stashed values')
                # HACK: here we are directly accessing the saved tensor outputs
                # for closed-over outputs so that they still have the grad_fn
                # from local autograd. Can we solve this more elegantly?
                kwargs = dict(kwargs)
                kwargs['stage_output'], kwargs['input_values'] = self.fwd_cache.pop(microbatch_id)

            if work_item.phase == Phase.FORWARD:
                self.outstanding += 1
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

            elif work_item.phase == Phase.BACKWARD:
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running backward')

                batch_id_to_remaining_backwards_stages[batch_id] -= 1

                if not isinstance(self.mod, torch.nn.parallel.distributed.DistributedDataParallel) or\
                        batch_id_to_remaining_backwards_stages[batch_id] == 0:
                    out_val = stage_backward(*args, **kwargs)
                else:
                    with self.mod.no_sync():
                        out_val = stage_backward(*args, **kwargs)

                # Schedule forward stage of a new micro-batch
                self.outstanding -= 1
            elif work_item.phase == Phase.SYNC_BARRIER:
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running sync_barrier')
                out_val = sync_barrier(*args, **kwargs)
            else:
                assert False, f'Unrecognized phase {work_item.phase} encountered in execution'

            logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Populating result of type {type(out_val)} '
                         f'for {key}')
            future.set_result(out_val)
            work_item.state = SchedState.DONE

            def name(ph, rank, mbid):
                phase_to_short_str = {
                    Phase.FORWARD: 'F',
                    Phase.BACKWARD: 'B',
                    Phase.SYNC_BARRIER: 'S',
                }
                return f"{phase_to_short_str[ph]}_{rank},{mbid}"

            name = name(work_item.phase, self.local_rank, work_item.microbatch_id)
            self.record_event(rank=self.local_rank, start_ts=start_ts, finish_ts=time.time(), id=name, name=name,
                              type=work_item.phase, mbid=work_item.microbatch_id)

    @rpc.functions.async_execution
    def invoke(self, unique_key : str, phase : Phase, args, kwargs, cur_microbatch : int, debug_str : str, batch_id : int, num_microbatches : int):
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
        future: torch.futures.Future = torch.futures.Future()
        # TODO: increase blocked_args_count for extra things like scheduling
        work_item = WorkItem(phase, args, kwargs, future, cur_microbatch, len(rref_args), {}, batch_id=batch_id,
                             num_microbatches=num_microbatches, debug_str=debug_str)
        logging.info(f'[{self.local_rank}][{cur_microbatch}] Invoke instantiated WorkItem {work_item} with key {unique_key}')
        if len(rref_args) == 0:
            # TODO: convert initial input into RRef?
            # We always put this work item into the ready queue, though we mark
            # it with different state flags depending on whether the schedule
            # would hold it based on max outstanding allowed
            work_item.set_trigger_state(self.max_outstanding)
            if work_item.state == SchedState.WAITING:
                logging.info(f'[{self.local_rank}][{cur_microbatch}] Schedule limits max outstanding micro-bactches. '
                             f'Initializing as WAITING workitem')
            else:
                logging.info(f'[{self.local_rank}][{cur_microbatch}] No RRef arguments. '
                             f'Scheduling directly as READY workitem')
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


        # Spawn asynchronous data transfers for each of the RRef arguments.
        _futures = []
        for arg_idx, rref_arg in enumerate(rref_args):
            logging.info(f'[{self.local_rank}][{cur_microbatch}] Launching asynchronous data transfer '
                         f'for RRef {arg_idx} {rref_arg}')
            self_rref: rpc.RRef = rpc.RRef(self)
            _futures.append(rpc.rpc_async(
                to=self.local_rank, func=async_transfer,
                args=(self.local_rank, cur_microbatch, self_rref, rref_arg, arg_idx, unique_key, self.max_outstanding)))

        if DEBUG:
            # Make exceptions visible
            for fut in _futures:
                fut.wait()

        return future

    def record_event(self, rank: int, start_ts: float, finish_ts: float, id: str, name: str, type: Optional[Any],
                     mbid: Optional[Any]):
        self.events.append(
            Event(rank=rank, start_ts=start_ts, finish_ts=finish_ts, id=id, name=name, type=type, mbid=mbid))

    def retrieve_events(self):
        events = self.events
        self.events = []
        return events


def get_grad_from_executor(executor, qualname):
    mod = executor.local_value().mod
    if isinstance(mod, torch.nn.parallel.DistributedDataParallel):
        mod = mod.module
    return mod.get_parameter(qualname).grad

def set_grad_in_executor(executor, qualname, value):
    mod = executor.local_value().mod
    if isinstance(mod, torch.nn.parallel.DistributedDataParallel):
        mod = mod.module
    param = mod.get_parameter(qualname)
    param.grad = value


class PipelineDriverBase:
    def __init__(self, pipe : Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size : int,
                 all_ranks : List[int] = None, _debug_mask_minibatches : bool = False,
                 dp_pg_cb=None):
        self.pipe = pipe
        self.world_size = world_size
        self.all_ranks = all_ranks
        self.executor_class = PipeStageExecutor
        self.args_chunk_spec = args_chunk_spec
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.output_chunk_spec = output_chunk_spec
        # Maximum outstanding micro-batches allowed by the pipeline schedule
        # None means no limit
        self.max_outstanding: Optional[int] = None
        self._debug_mask_minibatches = _debug_mask_minibatches
        self.dp_pg_cb = dp_pg_cb

    def _init_remote_executors(self):
        self.remote_stage_executor_rrefs : Dict[str, (int, torch.distributed.rpc.RRef)] = {}

        if self.all_ranks is not None:
            assert len(self.all_ranks) == self.world_size, "Explicitly specified ranks must match world_size"
        else:
            self.all_ranks = list(range(self.world_size))

        class ExecutorDescriptor:
            name : str
            mod : torch.nn.Module
            has_backward : bool = False

        split_gm = self.pipe.split_gm

        executor_descriptors = []
        bw_idx = -1
        for node in split_gm.graph.nodes:
            if node.op == 'call_module':
                target_mod = split_gm.get_submodule(node.target)
                descr = ExecutorDescriptor()
                descr.name = node.target
                descr.mod = target_mod
                executor_descriptors.append(descr)
            elif (node.op, node.target) == ('call_function', stage_backward):
                executor_descriptors[bw_idx].has_backward = True
                node.meta['fw_stage'] = executor_descriptors[bw_idx].name
                bw_idx -= 1

        assert all(d.has_backward for d in executor_descriptors) or all(not d.has_backward for d in executor_descriptors)

        if len(executor_descriptors) > self.world_size:
            raise RuntimeError(f'Tried to run pipeline with {len(executor_descriptors)} stages with a world size of '
                               f'{self.world_size}. Please ensure world_size is large enough to accommodate your pipeline.')

        if len(executor_descriptors) < self.world_size:
            warnings.warn(f'Running pipeline with {len(executor_descriptors)} stages on world_size of {self.world_size}. '
                          f'Remaining ranks will be idle.')

        pp_rank = 0
        for rank, descr in zip(self.all_ranks, executor_descriptors):
            kwargs = {'mod': descr.mod, 'local_rank': rank, 'max_outstanding': self.max_outstanding,
                      'dp_pg_cb': self.dp_pg_cb, 'pp_rank': pp_rank}
            self.remote_stage_executor_rrefs[descr.name] = (
                rank, rpc.remote(rank, self.executor_class, args=(), kwargs=kwargs))
            pp_rank += 1

    def run(self, chunks: int, *args, **kwargs):
        raise NotImplementedError('PipelineDriverBase is an abstract base class, please use a concrete '
                                  'implementation class.')


    def _sync_replicated_params(self):
        logging.info(f'[root] Synchronizing gradients for {len(self.pipe.replicated_params)} sets of replicated parameters')
        for param_set in self.pipe.replicated_params:
            grad_values = []
            for module_name, param_qualname in param_set.items():
                assert module_name in self.remote_stage_executor_rrefs
                rank, module_rref = self.remote_stage_executor_rrefs[module_name]
                grad_value = rpc.rpc_sync(rank, get_grad_from_executor, (module_rref, param_qualname))
                grad_values.append(grad_value)

            synced_value = torch.sum(torch.stack(grad_values), dim=0)

            for module_name, param_qualname in param_set.items():
                assert module_name in self.remote_stage_executor_rrefs
                rank, module_rref = self.remote_stage_executor_rrefs[module_name]
                rpc.rpc_sync(rank, set_grad_in_executor, (module_rref, param_qualname, synced_value))

    def _retrieve_output_values(self, microbatch_interpreters, last_nodes):
        logging.info(f'[root] Retrieving output values from {len(microbatch_interpreters)} chunks')
        output_vals = []
        for interp, last_node in zip(microbatch_interpreters, last_nodes):
            interp.run_until(lambda n : False)
            output_vals.append(interp.env[last_node])

        return torch.fx.node.map_aggregate(output_vals, to_here)

    def retrieve_events(self) -> List[Event]:
        events = []
        for descr_name, (rank, stage_executor) in self.remote_stage_executor_rrefs.items():
            events.extend(stage_executor.rpc_sync().retrieve_events())
        return events


class RemoteInterpreter(torch.fx.Interpreter):
    def __init__(self, remote_stage_executor_rrefs, module, cur_microbatch : int,   
                 args, kwargs, batch_id : int, num_microbatches : int, garbage_collect_values=True):
        super().__init__(module, garbage_collect_values)
        self.remote_stage_executor_rrefs = remote_stage_executor_rrefs
        self.cur_microbatch = cur_microbatch
        self.pc = 0
        self.node_list = list(self.module.graph.nodes)

        # Process args/kwargs

        # TODO: replace this with GraphModule.signature() when it lands
        parameters = []
        for node in self.module.graph.nodes:
            if node.op != 'placeholder':
                continue
            default = next(iter(node.args)) if node.args else Parameter.empty
            parameters.append(Parameter(node.name, Parameter.POSITIONAL_OR_KEYWORD, default=default))
        sig = Signature(parameters)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        self.args = bound_args.args
        self.args_iter = iter(self.args)
        self.batch_id = batch_id
        self.num_microbatches = num_microbatches

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        node = self.node_list[self.pc]

        if target in self.remote_stage_executor_rrefs:
            rank, stage_executor = self.remote_stage_executor_rrefs[target]
            logging.info(f'[root][{self.cur_microbatch}] Issuing {Phase.FORWARD} '
                         f'invocation for target {target} on rank {rank}')
            invocation_key = f'{self.cur_microbatch}_{node.name}'
            return stage_executor.remote().invoke(
                invocation_key, Phase.FORWARD, args, kwargs, self.cur_microbatch, debug_str=node.format_node(),
                batch_id=self.batch_id, num_microbatches=self.num_microbatches)
        else:
            logging.info(f'[root][{self.cur_microbatch}] Running local operation {target} from driver')
            return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], torch._C._distributed_rpc.PyRRef):
            return args[0].remote().__getitem__(args[1])
        elif target is stage_backward:
            node = self.node_list[self.pc]
            assert 'fw_stage' in node.meta
            rank, stage_executor = self.remote_stage_executor_rrefs[node.meta['fw_stage']]
            logging.info(f'[root][{self.cur_microbatch}] Issuing BW invocation '
                         f'for target {node.meta["fw_stage"]} on rank {rank}')
            invocation_key = f'{self.cur_microbatch}_{node.name}'
            return stage_executor.remote().invoke(
                invocation_key, Phase.BACKWARD, args, kwargs, self.cur_microbatch, debug_str=node.format_node(),
                batch_id=self.batch_id, num_microbatches=self.num_microbatches)
        elif target is sync_barrier:
            node = self.node_list[self.pc]
            # TODO: just assuming the last executor is indeed the executor for the
            # last stage. We should do this in a more principled way
            executor_keys = list(self.remote_stage_executor_rrefs.keys())
            rank, stage_executor = self.remote_stage_executor_rrefs[executor_keys[-1]]
            logging.info(f'[root][{self.cur_microbatch}] Issuing sync invocation '
                         f'on rank {rank}')
            invocation_key = f'{self.cur_microbatch}_{node.name}'
            return stage_executor.remote().invoke(
                invocation_key, Phase.SYNC_BARRIER, args, kwargs, self.cur_microbatch, debug_str=node.format_node(),
                batch_id=self.batch_id, num_microbatches=self.num_microbatches)
        else:
            raise AssertionError(f'Unknown operator {torch.typename(target)}')

    def run_until(self, predicate : Callable[[torch.fx.Node], bool]):
        for self.pc in range(self.pc, len(self.node_list)):
            node = self.node_list[self.pc]

            if predicate(node):
                return node

            # TODO: hoist run() implementation
            logging.info(f'[{self.cur_microbatch}] Issue command to run {node.format_node()}')
            self.env[node] = super().run_node(node)

            # TODO: we could potentially move this waiting to the use sites for an RRef
            # (i.e. during Interpreter.map_nodes_to_values or when we pass args/kwargs
            #  to the callees) as an optimization
            # TODO: is it possible for there to be a blocking version of this API?
            def wait_for_confirmation(n):
                if isinstance(n, torch._C._distributed_rpc.PyRRef):
                    while not n.confirmed_by_owner():
                        pass

            torch.fx.node.map_aggregate(self.env[node], wait_for_confirmation)

            if DEBUG and isinstance(self.env[node], torch._C._distributed_rpc.PyRRef):
                print(node, self.env[node])
                self.env[node].to_here()


class PipelineDriverFillDrain(PipelineDriverBase):
    def __init__(self, pipe: Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size: int,
                 all_ranks: List[int] = None, single_loss: bool = False, _debug_mask_minibatches: bool = False,
                 dp_pg_cb=None):
        super().__init__(pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size, all_ranks,
                         _debug_mask_minibatches, dp_pg_cb=dp_pg_cb)
        self.single_loss = single_loss

        self._init_remote_executors()

    def run(self, chunks: int, *args, **kwargs):
        if self.single_loss:
            raise NotImplementedError('Single minibatch loss not implemented')

        logging.info('[root] Running pipeline FillDrain')
        # Roadmap:
        # 1) Micro-batch splitting - divide input arguments out into concrete chunk values
        # 2) Interpreter tiling - one interpreter per micro-batch
        # 3) Scheduling - Use control logic to advance interpreters to issue round-robin
        #       forward work items, then round-robin losses, then round-robin backwards

        args_split, kwargs_split = split_args_kwargs_into_chunks(args, kwargs, self.args_chunk_spec,
                                                                 self.kwargs_chunk_spec, chunks,
                                                                 self._debug_mask_minibatches)

        microbatch_interpreters : List[RemoteInterpreter] = []

        batch_id = random.randint(0, 2**32)

        for chunk in range(chunks):
            logging.info(f'[root] Instantiating microbatch interpreter for chunk {chunk}')
            interp = RemoteInterpreter(self.remote_stage_executor_rrefs, self.pipe.split_gm, chunk, args_split[chunk],
                                       kwargs_split[chunk], batch_id=batch_id, num_microbatches=chunks)
            microbatch_interpreters.append(interp)

        logging.info(f'[root] {len(microbatch_interpreters)} instantiated')

        last_nodes = []
        for interp in microbatch_interpreters:
            last_nodes.append(interp.run_until(lambda n: n.op == 'output'))

        assert all(n == last_nodes[0] for n in last_nodes)
        assert last_nodes[0].op == 'output'

        local_results = self._retrieve_output_values(microbatch_interpreters, last_nodes)

        if self.pipe.has_loss_and_backwards:
            # Shared parameter sync
            # At this point, all of the gradient jobs should have been run
            # (by way of the synchronization dependency earlier)
            self._sync_replicated_params()

        return merge_chunks(local_results, self.output_chunk_spec, self._debug_mask_minibatches)


class PipelineDriver1F1B(PipelineDriverBase):
    def __init__(self, pipe: Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size: int,
                 all_ranks: List[int] = None, single_loss: bool = False, _debug_mask_minibatches: bool = False,
                 dp_pg_cb=None):
        super().__init__(pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size, all_ranks,
                         _debug_mask_minibatches, dp_pg_cb=dp_pg_cb)
        self.single_loss = single_loss
        # In 1F1B with backward stages, the maximum number of outstanding
        # micro-batches equals the number of pipeline stages
        if self.pipe.has_loss_and_backwards:
            self.max_outstanding = self.pipe.num_stages

        self._init_remote_executors()

    def run(self, chunks: int, *args, **kwargs):
        if self.single_loss:
            raise NotImplementedError('Single minibatch loss not implemented')

        logging.info('[root] Running pipeline 1F1B')

        args_split, kwargs_split = split_args_kwargs_into_chunks(args, kwargs, self.args_chunk_spec,
                                                                 self.kwargs_chunk_spec, chunks,
                                                                 self._debug_mask_minibatches)

        microbatch_interpreters : List[RemoteInterpreter] = []

        batch_id = random.randint(0, 2**32)

        for chunk in range(chunks):
            logging.info(f'[root] Instantiating microbatch interpreter for chunk {chunk}')
            interp = RemoteInterpreter(self.remote_stage_executor_rrefs, self.pipe.split_gm, chunk, args_split[chunk],
                                       kwargs_split[chunk], batch_id=batch_id, num_microbatches=chunks)
            microbatch_interpreters.append(interp)

        logging.info(f'[root] {len(microbatch_interpreters)} instantiated')

        last_nodes = []

        for i, interp in enumerate(microbatch_interpreters):
            logging.info(f'[root] Executing stages for chunk {i}')
            last_nodes.append(interp.run_until(lambda n: n.op == 'output'))

        local_results = self._retrieve_output_values(microbatch_interpreters, last_nodes)

        # There is backward pass, we should sync shared parameters
        if self.pipe.has_loss_and_backwards:
            self._sync_replicated_params()

        return merge_chunks(local_results, self.output_chunk_spec, self._debug_mask_minibatches)
