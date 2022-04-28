# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
import threading
import time
import warnings
from enum import Enum
from inspect import Parameter, Signature
from typing import Any, Callable, Dict, List, Tuple, Optional

import torch
import torch.distributed.rpc as rpc
import torch.fx

from pippy.IR import Pipe, stage_backward, sync_barrier, _null_coalesce_accumulate
from pippy.events import EventRecorder, EventsContext, Event, Allocator
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

class Phase(Enum):
    FORWARD = 0
    BACKWARD = 1
    ACCUMULATE_GRAD = 2
    SYNC_BARRIER = 3

# TODO: do we need this?
class SchedState(Enum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    DONE = 3


def event_name(ph, stage_id, mbid):
    phase_to_short_str = {
        Phase.FORWARD: 'F',
        Phase.BACKWARD: 'B',
        Phase.ACCUMULATE_GRAD: 'A',
        Phase.SYNC_BARRIER: 'S',
    }
    return f"{phase_to_short_str[ph]}_{stage_id},{mbid}"


def prev_event_name(ph: Any, all_stages: List[int], stage_id: int, mbid: Any):
    i = all_stages.index(stage_id)
    if ph == Phase.FORWARD and i > 0:
        prev_stage = all_stages[i - 1]
        return event_name(ph, prev_stage, mbid)
    elif ph == Phase.BACKWARD and i < len(all_stages) - 1:
        next_stage = all_stages[i + 1]
        return event_name(ph, next_stage, mbid)
    else:
        return None


def next_event_name(ph: Any, all_stages: List[int], stage_id: int, mbid: Any):
    i = all_stages.index(stage_id)
    if ph == Phase.FORWARD and i < len(all_stages) - 1:
        next_stage = all_stages[i + 1]
        return event_name(ph, next_stage, mbid)
    elif ph == Phase.BACKWARD and i > 0:
        prev_stage = all_stages[i - 1]
        return event_name(ph, prev_stage, mbid) if stage_id > 0 else None
    else:
        return None


class WorkItem:
    def __init__(
            self, stage_id, phase, args, kwargs, future, microbatch_id, blocked_args_count, ready_args,
            batch_id, num_microbatches, state=SchedState.WAITING, debug_str=''):
        args_to_fwd = ['stage_id', 'phase', 'args', 'kwargs', 'future', 'microbatch_id', 'blocked_args_count',
                       'ready_args', 'batch_id', 'num_microbatches', 'state', 'debug_str']

        for arg in args_to_fwd:
            setattr(self, arg, locals()[arg])

    stage_id : int
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


class ValueReference:
    def __init__(self, stage_id, unique_key):
        self.stage_id = stage_id
        self.unique_key = unique_key

    stage_id : int
    unique_key : str

    def __repr__(self):
        return f'ValueReference({self.stage_id}, {self.unique_key})'


class RefcountedFuture:
    future : torch.futures.Future
    refcount : int

    def __init__(self, future, refcount):
        self.future, self.refcount = future, refcount

    def release(self):
        """
        Decrement refcount by 1. Return True if this instance should be freed
        """
        assert self.refcount != 0, 'Detected reference counting inconsistency. Please report a bug to PiPPy'
        self.refcount -= 1
        return self.refcount == 0


class RankWorker(EventRecorder):
    """
    RankWorker is the underlying WorkItem processing engine for pipeline stages
    resident on this rank. WorkItems of multiple stages would share the same
    queue in the RankWorker. RankWorker will also maintain states like the
    number of outstanding WorkItems.

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
    """

    def __init__(self, local_rank, all_stages, max_outstanding=None, dp_pg_cb=None, pp_rank=None,
                 _record_mem_dumps=False):
        logging.info(f'[{local_rank}] Instantiating RankWorker')
        self.local_rank = local_rank
        self.all_stages = all_stages
        self.local_rank = local_rank
        self.dp_pg_cb = dp_pg_cb
        self.pp_rank = pp_rank
        self._record_mem_dumps = _record_mem_dumps

        # Maximum outstanding micro-batches of the pipeline schedule
        self.max_outstanding = max_outstanding
        # Keeps track of the outstanding micro-batches in current rank executor
        self.outstanding = 0
        self.stage_executors : Dict[int, PipeStageExecutor] = {}
        self.events: List[Event] = []

        self.waiting_runlist_lock = threading.Lock()
        # self.waiting_runlist (*and the contained WorkItems*) are guarded by
        # self.waiting_runlist_lock
        self.waiting_runlist : Dict[str, WorkItem] = {}

        self.ready_runlist_lock = threading.Lock()
        self.ready_runlist_cv = threading.Condition(self.ready_runlist_lock)
        self.ready_runlist : Dict[str, WorkItem] = {}

        self.worker_thread = threading.Thread(target=self.worker_loop,
                                              name=f'worker_{self.local_rank}', daemon=True)
        self.worker_thread.start()


    def create_stage_executor(self, stage_id, mod):
        if stage_id in self.stage_executors:
            raise AssertionError(f'Rank {self.local_rank} already has stage {stage_id}')
        self.stage_executors[stage_id] = PipeStageExecutor(stage_id=stage_id,
                                                           mod=mod, rank_worker=self,
                                                           dp_pg_cb=self.dp_pg_cb,
                                                           pp_rank=self.pp_rank)
        return self.stage_executors[stage_id]

    def enqueue_ready_runlist(self, unique_key, work_item):
        with self.ready_runlist_cv:
            logging.info(f'[{self.local_rank}] Current ready runlist keys: {self.ready_runlist.keys()}')
            self.ready_runlist[unique_key] = work_item
            self.ready_runlist_cv.notify()

    def enqueue_waiting_runlist(self, unique_key, work_item):
        with self.waiting_runlist_lock:
            logging.info(f'[{self.local_rank}] Current waiting runlist keys: {self.waiting_runlist.keys()}')
            assert unique_key not in self.waiting_runlist, f'key {unique_key} already in waiting runlist {self.waiting_runlist}'
            self.waiting_runlist[unique_key] = work_item

    def worker_loop(self):
        batch_id_to_remaining_backward_microbatches : Dict[int, int] = {}
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
                    if (self.ready_runlist[key].phase == Phase.FORWARD and
                            self.max_outstanding is not None and
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
            args_value_refs = work_item.args
            kwargs_value_refs = work_item.kwargs
            future = work_item.future
            microbatch_id = work_item.microbatch_id
            ready_args = work_item.ready_args
            phase = work_item.phase
            try:
                stage_executor = self.stage_executors[work_item.stage_id]
            except KeyError:
                raise RuntimeError(f'Rank {self.local_rank} does not have stage {work_item.stage_id}'
                                   f'Current keys {self.stage_executors.keys()}')

            batch_id = work_item.batch_id
            num_microbatches = work_item.num_microbatches

            if batch_id not in batch_id_to_remaining_backward_microbatches:
                batch_id_to_remaining_backward_microbatches[batch_id] = num_microbatches

            start_ts = time.time()
            name = event_name(work_item.phase, work_item.stage_id, work_item.microbatch_id)
            if self._record_mem_dumps:
                for peer_executor_rref in stage_executor.peer_executors.values():
                    peer_executor_rref.rpc_sync()._record_dump(f'M{name}_start', start_ts)

            value_ref_arg_idx = 0

            def retrieve_value_ref_args_by_idx(a):
                if isinstance(a, ValueReference):
                    nonlocal value_ref_arg_idx
                    val = ready_args[value_ref_arg_idx]
                    value_ref_arg_idx += 1
                    return val
                else:
                    return a

            args = torch.fx.node.map_aggregate(args_value_refs, retrieve_value_ref_args_by_idx)
            kwargs = torch.fx.node.map_aggregate(kwargs_value_refs, retrieve_value_ref_args_by_idx)

            if phase == Phase.BACKWARD:
                logging.info(
                    f'[{self.local_rank}][{work_item.microbatch_id}] Running backward phase. Retrieving stashed values')
                # HACK: here we are directly accessing the saved tensor outputs
                # for closed-over outputs so that they still have the grad_fn
                # from local autograd. Can we solve this more elegantly?
                kwargs = dict(kwargs)
                kwargs['stage_output'], kwargs['input_values'] = stage_executor.fwd_cache.pop(microbatch_id)

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

                if isinstance(stage_executor.mod, torch.nn.parallel.distributed.DistributedDataParallel):
                    with stage_executor.mod.no_sync():
                        out_val = stage_executor.mod(*args, **kwargs)
                else:
                    out_val = stage_executor.mod(*args, **kwargs)

                stage_executor.fwd_cache[microbatch_id] = (out_val if isinstance(out_val, tuple) else (out_val,), flat_tensor_args)

            elif work_item.phase == Phase.BACKWARD:
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running backward')

                batch_id_to_remaining_backward_microbatches[batch_id] -= 1

                if isinstance(stage_executor.mod, torch.nn.parallel.distributed.DistributedDataParallel) and \
                        batch_id_to_remaining_backward_microbatches[batch_id] == 0:
                    # HACK: reaching into DDP implementation details here. Is there a better way?
                    stage_executor.mod.reducer.prepare_for_backward(
                        list(torch.nn.parallel.distributed._find_tensors(kwargs['stage_output'])))

                out_val = stage_backward(*args, **kwargs)

                # Schedule forward stage of a new micro-batch
                self.outstanding -= 1
            elif work_item.phase == Phase.ACCUMULATE_GRAD:
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running accumulate grad')
                out_val = _null_coalesce_accumulate(*args, **kwargs)
            elif work_item.phase == Phase.SYNC_BARRIER:
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Running sync_barrier')
                out_val = sync_barrier(*args, **kwargs)
            else:
                assert False, f'Unrecognized phase {work_item.phase} encountered in execution'

            logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] Populating result of type {type(out_val)} '
                         f'for {key}')
            future.set_result(out_val)
            work_item.state = SchedState.DONE

            prev_name = prev_event_name(work_item.phase, self.all_stages, work_item.stage_id, work_item.microbatch_id)
            next_name = next_event_name(work_item.phase, self.all_stages, work_item.stage_id, work_item.microbatch_id)
            finish_ts = time.time()
            self.record_event(rank=self.local_rank, start_ts=start_ts, finish_ts=finish_ts, id=name,
                              name=name, type=work_item.phase, mbid=work_item.microbatch_id)
            self.record_event_dependency(from_id=prev_name, to_id=name, type='transfer')
            self.record_event_dependency(from_id=name, to_id=next_name, type='transfer')

            if self._record_mem_dumps:
                for peer_executor_rref in stage_executor.peer_executors.values():
                    peer_executor_rref.rpc_sync()._record_dump(f'M{name}_finish', finish_ts)

    # For work item marked with runlist_key, update its operand list with value
    def update_run_list(self, runlist_key, arg_idx, value):
        with self.waiting_runlist_lock:
            work_item = self.waiting_runlist[runlist_key]
            work_item.ready_args[arg_idx] = value
            work_item.blocked_args_count -= 1
            if work_item.blocked_args_count == 0:
                with self.ready_runlist_cv:
                    work_item.state = SchedState.READY
                    self.ready_runlist[runlist_key] = self.waiting_runlist.pop(runlist_key)
                    self.ready_runlist_cv.notify()
                logging.info(f'[{self.local_rank}][{work_item.microbatch_id}] all operands ready')


class PipeStageExecutor(EventRecorder):
    """
    PipeStageExecutor encapsulates the execution semantics of a fragment of
    code on a pipeline stage. PipeStageExecutor handles:

    * Ownership of the stage's module and its recursive submodules/parameters
    * Serving as an entrypoint for the driver to push jobs into RankWorker's queue
    * TODO: gradient checkpointing
    """

    def __init__(self, stage_id, mod, rank_worker, dp_pg_cb, pp_rank):
        logging.info(f'Instantiating PipeStageExecutor for stage {stage_id}')
        self.stage_id = stage_id
        if dp_pg_cb is not None:
            assert pp_rank is not None
            self.mod = torch.nn.parallel.DistributedDataParallel(mod, process_group=dp_pg_cb(pp_rank))
        else:
            self.mod = mod
        self.rank_worker = rank_worker
        # map microbatch ID to list of forward tensor args
        self.fwd_cache : Dict[int, Tuple[Any, List[torch.Tensor]]] = {}

        self.value_store_lock = threading.Lock()
        self.value_store : Dict[str, RefcountedFuture] = {}

        self.peer_executors : Dict[int, torch._C._distributed_rpc.PyRRef] = None

    def __getstate__(self):
        # Adding an empty __getstate__ function here to work around the DDP pickling issue (#153) that occurs when the
        # PipelineDiver asks PipeStageExecutors to install_peer_executor(a list of RRefs)
        # More elegant solution is needed in CUDAFuture or RPC to avoid pickling when users do not need to transfer
        # tensors
        pass

    def install_peer_executors(self, peer_executors):
        assert self.peer_executors is None
        self.peer_executors = peer_executors
        return None

    def invoke(self, output_unique_key : str, phase : Phase, args, kwargs, cur_microbatch : int, debug_str : str,
               output_refcount : int, batch_id : int, num_microbatches : int):
        ts = time.time()
        forward_name = event_name(Phase.FORWARD, self.stage_id, cur_microbatch)
        name = f"R{forward_name}"
        self.record_event(rank=self.rank_worker.local_rank, start_ts=ts, finish_ts=ts, id=name, name=name, type='received', mbid=cur_microbatch)
        self.record_event_dependency(from_id=name, to_id=forward_name, type='waiting')
        # TODO: do we need to serialize calls to invoke() to preserve the order in which WorkItems appear for
        # static schedules?

        logging.info(f'[{self.stage_id}][{cur_microbatch}] Received invoke call for {debug_str}')
        # Extract all ValueRef arguments so we can spawn asynchronous data transfers
        # for each of them
        value_ref_args : List[ValueReference] = []

        def extract_value_ref_args(arg):
            if isinstance(arg, ValueReference):
                value_ref_args.append(arg)
        torch.fx.node.map_aggregate(args, extract_value_ref_args)
        torch.fx.node.map_aggregate(kwargs, extract_value_ref_args)

        logging.info(f'[{self.stage_id}][{cur_microbatch}] Invoke call found {len(value_ref_args)} ValueReference arguments')

        # Construct WorkItem for this microbatch+phase and record it in the
        # waiting runlist

        # We provide device to the Future constructor so that between
        # future.set_result() and future.wait() correct dependencies can be
        # captured
        # We assume the output value is on the same device as the stage's
        # module, and that all parameters in the module are on the same device
        # HACK: we assume the module has at least one parameter
        param = next(self.mod.parameters(), None)
        if param is None:
            warnings.warn(f"Module of stage {self.stage_id} has 0 parameters, "
                          f"cannot figure out device. Setting it to cpu")
        else:
            device = param.device

        # Future constructor does not accept CPU device, must set to None
        future: torch.futures.Future = torch.futures.Future(devices=None if
                                                            param is None or device.type == 'cpu'
                                                            else [device])
        # TODO: increase blocked_args_count for extra things like scheduling
        work_item = WorkItem(stage_id=self.stage_id, phase=phase, args=args, kwargs=kwargs, future=future,
                             microbatch_id=cur_microbatch, blocked_args_count=len(value_ref_args), ready_args={},
                             batch_id=batch_id, num_microbatches=num_microbatches, debug_str=debug_str)
        logging.info(f'[{self.stage_id}][{cur_microbatch}] Invoke instantiated WorkItem {work_item} with key {output_unique_key}')
        if len(value_ref_args) == 0:
            # TODO: convert initial input into ValueRef?
            # We always put this work item into the ready queue, though we mark
            # it with different state flags depending on whether the schedule
            # would hold it based on max outstanding allowed
            work_item.state = SchedState.READY
            logging.info(f'[{self.stage_id}][{cur_microbatch}] No RRef arguments. '
                         f'Scheduling directly as READY workitem')
            self.rank_worker.enqueue_ready_runlist(output_unique_key, work_item)
        else:
            logging.info(f'[{self.stage_id}][{cur_microbatch}] Scheduling WorkItem as WAITING workitem')
            work_item.state = SchedState.WAITING
            self.rank_worker.enqueue_waiting_runlist(output_unique_key, work_item)

        # Spawn asynchronous data transfers for each of the ValueRef arguments.
        _futures = []
        for arg_idx, value_ref_arg in enumerate(value_ref_args):
            logging.info(f'[{self.stage_id}][{cur_microbatch}] Launching asynchronous data transfer for '
                         f'ValueReference {arg_idx} {value_ref_arg}')
            assert self.peer_executors is not None
            _futures.append(self.async_transfer(cur_microbatch, value_ref_arg,
                                                arg_idx, output_unique_key))

        if DEBUG:
            # Make exceptions visible
            for fut in _futures:
                fut.wait()

        with self.value_store_lock:
            assert output_unique_key not in self.value_store
            self.value_store[output_unique_key] = RefcountedFuture(future, output_refcount)

        return ValueReference(self.stage_id, output_unique_key)

    def index_value(self, output_unique_key : str, output_refcount : int, value_ref, idx):
        # For the purposes of refcounting, decrement this use
        with self.value_store_lock:
            refcounted_future = self.value_store[value_ref.unique_key]
            if refcounted_future.release():
                self.value_store.pop(value_ref.unique_key)

            indexed = refcounted_future.future.then(lambda f: f.value()[idx])

            self.value_store[output_unique_key] = RefcountedFuture(indexed, output_refcount)

        return ValueReference(self.stage_id, output_unique_key)

    def get_value(self, caller_stage, runlist_key, microbatch, value_ref_arg):
        callee_stage = value_ref_arg.stage_id
        logging.info(f'[{callee_stage}][{microbatch}] Executing async transfer of value '
                     f'{value_ref_arg} initiated by stage {caller_stage} for {runlist_key}')
        assert callee_stage == self.stage_id, "Mismatch between ValueRef and stage executor"

        with self.value_store_lock:
            refcounted_future = self.value_store[value_ref_arg.unique_key]

        value = refcounted_future.future.wait()

        with self.value_store_lock:
            if refcounted_future.release():
                self.value_store.pop(value_ref_arg.unique_key)

        return value

    def async_transfer(self, microbatch, value_ref_arg, arg_idx, runlist_key):
        logging.info(f'[{self.stage_id}][{microbatch}] Starting transfer')
        value_ref_executor_rref = self.peer_executors[value_ref_arg.stage_id]
        fut = value_ref_executor_rref.rpc_async(timeout=0).get_value(
            self.stage_id, runlist_key, microbatch, value_ref_arg)

        def bottom_half(fut):
            logging.info(f'[{self.stage_id}][{microbatch}] Completing transfer of value {value_ref_arg} '
                         f'for runlist item {runlist_key}')
            value = fut.value()
            self.rank_worker.update_run_list(runlist_key, arg_idx, value)

        return fut.then(bottom_half)

    def get_grad(self, qualname):
        mod = self.mod
        if isinstance(mod, torch.nn.parallel.DistributedDataParallel):
            mod = mod.module
        return mod.get_parameter(qualname).grad

    def set_grad(self, qualname, value):
        mod = self.mod
        if isinstance(mod, torch.nn.parallel.DistributedDataParallel):
            mod = mod.module
        param = mod.get_parameter(qualname)
        param.grad = value

    def train(self, mode=True):
        self.mod.train(mode=mode)

    def _should_instantiate_optim(self):
        return len(list(self.mod.parameters())) > 0

    def instantiate_optimizer(self, optim_class, *args, **kwargs):
        assert self._should_instantiate_optim()
        return optim_class(self.mod.parameters(), *args, **kwargs)

    def _record_dump(self, dump_id, ts):
        first_param = next(self.mod.parameters(), None)
        device: torch.device = first_param.device if first_param is not None else torch.device('cpu')
        if device.type == "cuda":
            self.record_dump(rank=self.rank_worker.local_rank, ts=ts, id=dump_id, name=dump_id, type='dump',
                             allocators={"CUDA": Allocator(f"{self.rank_worker.local_rank}", {
                                 "size": int(torch.cuda.memory_allocated()),
                                 "memory_allocated": int(torch.cuda.memory_allocated()),
                                 "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                                 "memory_reserved": int(torch.cuda.memory_reserved()),
                                 "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                             })})


def _wait_for_all(rpc_futs):
    # Stolen from DistributedOptimizer implementation
    # TODO: improve error propagation
    exception = None
    results = []
    for fut in rpc_futs:
        try:
            results.append(fut.wait())
        except Exception as e:
            results.append(e)
            exception = e
    if exception is not None:
        raise exception
    return results


class PipelineOptimizer:
    def __init__(self, remote_optims):
        self.remote_optims = remote_optims

    def zero_grad(self, set_to_none : bool = False):
        futs = []
        for optim in self.remote_optims:
            futs.append(optim.rpc_async().zero_grad(set_to_none))
        _wait_for_all(futs)

    def step(self, closure=None):
        futs = []
        for optim in self.remote_optims:
            futs.append(optim.rpc_async().step(closure))
        _wait_for_all(futs)


class PipelineDriverBase:
    def __init__(self, pipe : Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size : int,
                 all_ranks : List[int] = None, _debug_mask_minibatches : bool = False,
                 dp_pg_cb=None, max_outstanding=None, interleave_stages=False, _record_mem_dumps=False):
        self.pipe = pipe
        self.world_size = world_size
        self.all_ranks = all_ranks
        self.args_chunk_spec = args_chunk_spec
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.output_chunk_spec = output_chunk_spec
        # Maximum outstanding micro-batches allowed by the pipeline schedule
        # None means no limit
        self.max_outstanding: Optional[int] = max_outstanding
        self._debug_mask_minibatches = _debug_mask_minibatches
        self.interleave_stages = interleave_stages

        self.microbatch_interpreters: List[RemoteInterpreter] = []
        self.dp_pg_cb = dp_pg_cb
        self.batch_id = 0
        self._record_mem_dumps = _record_mem_dumps

    def _init_remote_executors(self):
        self.rank_worker_rrefs : Dict[int, torch.distributed.rpc.RRef] = {}
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
            elif (node.op, node.target) == ('call_function', _null_coalesce_accumulate):
                node.meta['fw_stage'] = executor_descriptors[bw_idx].name

        assert all(d.has_backward for d in executor_descriptors) or all(not d.has_backward for d in executor_descriptors)

        if len(executor_descriptors) > self.world_size:
            if not self.interleave_stages:
                raise RuntimeError(f'Tried to run pipeline with {len(executor_descriptors)} stages with a world size of '
                                   f'{self.world_size}. Please ensure world_size is large enough to accommodate your pipeline.')

        ranks_to_launch = self.world_size
        n_stages = len(executor_descriptors)
        if n_stages < self.world_size:
            ranks_to_launch = n_stages
            warnings.warn(f'Running pipeline with {n_stages} stages on world_size of {self.world_size}. '
                          f'Remaining ranks will be idle.')

        if self.interleave_stages and n_stages <= ranks_to_launch:
            self.interleave_stages = False
            warnings.warn('Falling back from Interleaved 1F1B to 1F1B '
                          'since there are enough ranks to support one stage per rank')

        # Fire up rank workers
        all_stages = list(range(n_stages))
        pp_rank = 0
        for rank in self.all_ranks[:ranks_to_launch]:
            kwargs = {'local_rank': rank,
                      'all_stages': all_stages,
                      'max_outstanding': self.max_outstanding,
                      'dp_pg_cb': self.dp_pg_cb,
                      'pp_rank': pp_rank,
                      '_record_mem_dumps': self._record_mem_dumps}
            self.rank_worker_rrefs[rank] = rpc.remote(rank, RankWorker, args=(), kwargs=kwargs)
            pp_rank += 1

        self.stage_to_executor : Dict = {}

        for stage_id, descr in enumerate(executor_descriptors):
            # Assign stages to rank workers in a round-robin fashion
            rank = self.all_ranks[stage_id % self.world_size]
            self.remote_stage_executor_rrefs[descr.name] = (stage_id,
                                                            self.rank_worker_rrefs[rank].remote().create_stage_executor(
                                                                stage_id=stage_id,
                                                                mod=descr.mod))
            self.stage_to_executor[stage_id] = self.remote_stage_executor_rrefs[descr.name][1]

        # Inform executors of their peers
        for stage_id, executor in self.stage_to_executor.items():
            executor.rpc_sync().install_peer_executors(self.stage_to_executor)

    def run(self, chunks: int, *args, **kwargs):
        raise NotImplementedError('PipelineDriverBase is an abstract base class, please use a concrete '
                                  'implementation class.')

    def train(self, mode=True):
        for executor in self.stage_to_executor.values():
            executor.rpc_sync().train(mode=mode)

    def eval(self):
        self.train(mode=False)

    def instantiate_optimizer(self, optim_class, *args, **kwargs):
        remote_optims = []
        for executor in self.stage_to_executor.values():
            if executor.rpc_sync()._should_instantiate_optim():
                remote_optim = executor.remote().instantiate_optimizer(optim_class, *args, **kwargs)
                remote_optims.append(remote_optim)

        return PipelineOptimizer([optim for optim in remote_optims if optim is not None])

    def _sync_replicated_params(self):
        logging.info(f'[root] Synchronizing gradients for {len(self.pipe.replicated_params)} sets of replicated parameters')
        for param_set in self.pipe.replicated_params:
            grad_values = []
            for module_name, param_qualname in param_set.items():
                assert module_name in self.remote_stage_executor_rrefs
                stage_id, module_rref = self.remote_stage_executor_rrefs[module_name]
                grad_value = module_rref.rpc_sync().get_grad(param_qualname)
                grad_values.append(grad_value)

            synced_value = torch.sum(torch.stack(grad_values), dim=0)

            for module_name, param_qualname in param_set.items():
                assert module_name in self.remote_stage_executor_rrefs
                stage_id, module_rref = self.remote_stage_executor_rrefs[module_name]
                module_rref.rpc_sync().set_grad(param_qualname, synced_value)

    def _retrieve_output_values(self, microbatch_interpreters, last_nodes):
        logging.info(f'[root] Retrieving output values from {len(microbatch_interpreters)} chunks')
        output_vals = []
        for interp, last_node in zip(microbatch_interpreters, last_nodes):
            interp.run_until(lambda n : False)
            output_vals.append(interp.env[last_node])

        # First kick of async transfers to retrieve ValueReference values
        def initiate_async_transfer(a):
            if isinstance(a, ValueReference):
                value_ref_executor_rref = self.stage_to_executor[a.stage_id]
                return value_ref_executor_rref.rpc_async(timeout=0).get_value(
                    'root', 'collect', -1, a)
            else:
                return a

        output_vals = torch.fx.node.map_aggregate(output_vals, initiate_async_transfer)

        # Then wait for futures to be ready
        return torch.fx.node.map_aggregate(output_vals, lambda a: a.wait() if isinstance(a, torch._C.Future) else a)

    def retrieve_events(self) -> EventsContext:
        events_context = EventsContext()
        for rank, worker_rref in self.rank_worker_rrefs.items():
            events_context.update(worker_rref.rpc_sync().retrieve_events())
        for interp in self.microbatch_interpreters:
            events_context.update(interp.retrieve_events())
        for _, executor_rref in self.remote_stage_executor_rrefs.values():
            events_context.update(executor_rref.rpc_sync().retrieve_events())
        events_context.events.sort(key=lambda e: e.start_ts)
        return events_context


class RemoteInterpreter(torch.fx.Interpreter, EventRecorder):
    def __init__(self, remote_stage_executor_rrefs, stage_to_executor, module, cur_microbatch : int,
                 args, kwargs, batch_id : int, num_microbatches : int, garbage_collect_values=True):
        super().__init__(module, garbage_collect_values)
        self.remote_stage_executor_rrefs = remote_stage_executor_rrefs
        self.stage_to_executor = stage_to_executor
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
            stage_id, stage_executor = self.remote_stage_executor_rrefs[target]
            logging.info(f'[root][{self.cur_microbatch}] Issuing {Phase.FORWARD} '
                         f'invocation for target {target} on stage {stage_id}')
            invocation_key = f'{self.cur_microbatch}_{node.name}'
            ts = time.time()
            forward_name = event_name(Phase.FORWARD, stage_id, self.cur_microbatch)
            name = f"I{forward_name}"
            self.record_event(rank=0, start_ts=ts, finish_ts=ts, id=name, name=name, type='invoke',
                              mbid=self.cur_microbatch)
            self.record_event_dependency(from_id=name, to_id=f"R{forward_name}", type='invoke')
            return stage_executor.rpc_sync().invoke(
                invocation_key, Phase.FORWARD, args, kwargs, self.cur_microbatch, debug_str=node.format_node(),
                output_refcount=len(node.users), batch_id=self.batch_id, num_microbatches=self.num_microbatches)
        else:
            logging.info(f'[root][{self.cur_microbatch}] Running local operation {target} from driver')
            return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        node = self.node_list[self.pc]
        invocation_key = f'{self.cur_microbatch}_{node.name}'
        if target is operator.getitem and isinstance(args[0], ValueReference):
            stage_executor = self.stage_to_executor[args[0].stage_id]
            return stage_executor.rpc_sync(timeout=0).index_value(
                output_unique_key=invocation_key, value_ref=args[0], output_refcount=len(node.users),
                idx=args[1])
        elif target is stage_backward:
            assert 'fw_stage' in node.meta
            stage_id, stage_executor = self.remote_stage_executor_rrefs[node.meta['fw_stage']]
            logging.info(f'[root][{self.cur_microbatch}] Issuing BW invocation '
                         f'for target {node.meta["fw_stage"]} on stage {stage_id}')
            ts = time.time()
            backward_name = event_name(Phase.BACKWARD, stage_id, self.cur_microbatch)
            name = f"I{backward_name}"
            self.record_event(rank=0, start_ts=ts, finish_ts=ts, id=name, name=name, type='invoke',
                              mbid=self.cur_microbatch)
            self.record_event_dependency(from_id=name, to_id=backward_name, type='invoke')
            return stage_executor.rpc_sync().invoke(
                invocation_key, Phase.BACKWARD, args, kwargs, self.cur_microbatch, debug_str=node.format_node(),
                output_refcount=len(node.users), batch_id=self.batch_id, num_microbatches=self.num_microbatches)
        elif target is sync_barrier:
            executor_keys = list(self.remote_stage_executor_rrefs.keys())
            stage_id, stage_executor = self.remote_stage_executor_rrefs[executor_keys[0]]
            logging.info(f'[root][{self.cur_microbatch}] Issuing sync invocation '
                         f'on stage {stage_id}')
            return stage_executor.rpc_sync().invoke(
                invocation_key, Phase.SYNC_BARRIER, args, kwargs, self.cur_microbatch, debug_str=node.format_node(),
                output_refcount=len(node.users), batch_id=self.batch_id, num_microbatches=self.num_microbatches)
        elif target is _null_coalesce_accumulate:
            assert 'fw_stage' in node.meta
            stage_id, stage_executor = self.remote_stage_executor_rrefs[node.meta['fw_stage']]
            logging.info(f'[root][{self.cur_microbatch}] Issuing accumulate grad invocation '
                         f'for target {node.meta["fw_stage"]} on stage {stage_id}')
            return stage_executor.rpc_sync().invoke(
                invocation_key, Phase.ACCUMULATE_GRAD, args, kwargs, self.cur_microbatch,
                debug_str=node.format_node(),
                output_refcount=len(node.users), batch_id=self.batch_id, num_microbatches=self.num_microbatches)
        else:
            raise AssertionError(f'Unknown operator {torch.typename(target)}')

    def run_until(self, predicate: Callable[[torch.fx.Node], bool]):
        while self.pc < len(self.node_list):
            node = self.node_list[self.pc]

            if predicate(node):
                return node

            return self.run_one(node)

    def run_one(self, node):
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

        self.pc += 1
        return node


class PipelineDriverFillDrain(PipelineDriverBase):
    def __init__(self, pipe: Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size: int,
                 all_ranks: List[int] = None, single_loss: bool = False, _debug_mask_minibatches: bool = False,
                 dp_pg_cb=None, max_outstanding=None, interleave_stages=False, _record_mem_dumps=False):
        super().__init__(pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size, all_ranks,
                         _debug_mask_minibatches, dp_pg_cb=dp_pg_cb, max_outstanding=max_outstanding,
                         interleave_stages=interleave_stages, _record_mem_dumps=_record_mem_dumps)
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

        self.microbatch_interpreters = []

        batch_id = self.batch_id
        self.batch_id += 1

        for chunk in range(chunks):
            logging.info(f'[root] Instantiating microbatch interpreter for chunk {chunk}')
            interp = RemoteInterpreter(remote_stage_executor_rrefs=self.remote_stage_executor_rrefs,
                                       stage_to_executor=self.stage_to_executor, module=self.pipe.split_gm,
                                       cur_microbatch=chunk, args=args_split[chunk], kwargs=kwargs_split[chunk],
                                       batch_id=batch_id, num_microbatches=chunks)
            self.microbatch_interpreters.append(interp)

        logging.info(f'[root] {len(self.microbatch_interpreters)} instantiated')

        # Deterministic clock cycle - see torchgpipe paper section 3.2.1 for details

        # Advance past placeholders
        for interp in self.microbatch_interpreters:
            interp.run_until(lambda n: n.op != 'placeholder')

        # Ramp-up, admit diagonal wavefront until we get to a full diagonal
        # location in the matrix
        for ramp_up_idx in range(len(self.microbatch_interpreters)):
            for i in range(ramp_up_idx + 1):
                interp = self.microbatch_interpreters[i]
                start_node = interp.node_list[interp.pc]

                def run_including_indexing(n):
                    # Run the node we start with including all nodes that are tuple
                    # indexing, then stop
                    return n != start_node and n.target != operator.getitem

                interp.run_until(run_including_indexing)

        # Steady-state. We have a full diagonal in the matrix; keep dispatching
        # across the diagonal

        any_valid = True
        while any_valid:
            any_valid = False
            for interp in self.microbatch_interpreters:
                start_node = interp.node_list[interp.pc]

                def run_including_indexing(n):
                    # Run the node we start with including all nodes that are
                    # tuple indexing, but also stop at the output node. Because
                    # we are on a diagonal, interpreters as lower microbatch IDs
                    # will be invoked when their pc's point to the output node
                    # multiple times.
                    if n.op == 'output':
                        return True

                    return n != start_node and n.target != operator.getitem

                interp.run_until(run_including_indexing)

                any_valid |= interp.node_list[interp.pc] != start_node


        last_nodes = [interp.node_list[interp.pc] for interp in self.microbatch_interpreters]
        assert all(node.op == 'output' for node in last_nodes)

        local_results = self._retrieve_output_values(self.microbatch_interpreters, last_nodes)

        if self.pipe.has_loss_and_backwards:
            # Shared parameter sync
            # At this point, all of the gradient jobs should have been run
            # (by way of the synchronization dependency earlier)
            self._sync_replicated_params()

        return merge_chunks(local_results, self.output_chunk_spec, self._debug_mask_minibatches)


class PipelineDriver1F1B(PipelineDriverFillDrain):
    def __init__(self, pipe: Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size: int,
                 all_ranks: List[int] = None, single_loss: bool = False, _debug_mask_minibatches: bool = False,
                 dp_pg_cb=None, interleave_stages=False, _record_mem_dumps=False):
        # In 1F1B with backward stages, the maximum number of outstanding
        # micro-batches equals the number of pipeline stages
        max_outstanding = pipe.num_stages if pipe.has_loss_and_backwards else None

        super().__init__(pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size, all_ranks,
                         single_loss, _debug_mask_minibatches, dp_pg_cb=dp_pg_cb, max_outstanding=max_outstanding,
                         interleave_stages=interleave_stages, _record_mem_dumps=_record_mem_dumps)

class PipelineDriverInterleaved1F1B(PipelineDriver1F1B):
    def __init__(self, pipe : Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size : int,
                 all_ranks : List[int] = None, single_loss : bool = False, _debug_mask_minibatches: bool = False,
                 dp_pg_cb=None, _record_mem_dumps=False):
        super().__init__(pipe, args_chunk_spec, kwargs_chunk_spec,
                         output_chunk_spec, world_size, all_ranks, single_loss,
                         _debug_mask_minibatches, dp_pg_cb=dp_pg_cb, interleave_stages=True,
                         _record_mem_dumps=_record_mem_dumps)
