# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
import threading
import time
import warnings
from enum import Enum
from inspect import Parameter, Signature
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed.rpc as rpc

import pippy.fx
from pippy.backward import (
    _null_coalesce_accumulate,
    stage_backward,
    sync_barrier,
)
from pippy.events import Allocator, Event, EventRecorder, EventsContext
from pippy.fx.passes import shape_prop

from pippy.IR import Pipe
from pippy.microbatch import (
    gen_output_chunk_spec,
    LossReducer,
    merge_chunks,
    split_args_kwargs_into_chunks,
    sum_reducer,
)
from pippy.utils import flatten_args_detach

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
        Phase.FORWARD: "F",
        Phase.BACKWARD: "B",
        Phase.ACCUMULATE_GRAD: "A",
        Phase.SYNC_BARRIER: "S",
    }
    return f"{phase_to_short_str[ph]}_{stage_id},{mbid}"


def event_id(ph, stage_id, mbid, bid):
    return f"{event_name(ph, stage_id, mbid)},{bid}"


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
        self,
        stage_id,
        phase,
        args,
        kwargs,
        future,
        microbatch_id,
        blocked_args_count,
        ready_args,
        batch_id,
        num_microbatches,
        state=SchedState.WAITING,
        debug_str="",
    ):
        args_to_fwd = [
            "stage_id",
            "phase",
            "args",
            "kwargs",
            "future",
            "microbatch_id",
            "blocked_args_count",
            "ready_args",
            "batch_id",
            "num_microbatches",
            "state",
            "debug_str",
        ]

        for arg in args_to_fwd:
            setattr(self, arg, locals()[arg])

    stage_id: int
    phase: Phase
    args: Tuple[Any]
    kwargs: Dict[str, Any]
    future: torch.futures.Future
    microbatch_id: int

    blocked_args_count: int
    ready_args: Dict[int, Any]
    state: SchedState
    debug_str: str

    batch_id: int
    num_microbatches: int

    def __str__(self):
        return f"WorkItem({self.debug_str})"


class ValueReference:
    def __init__(self, stage_id, unique_key):
        self.stage_id = stage_id
        self.unique_key = unique_key
        self.meta: Dict[str, Any] = {}

    stage_id: int
    unique_key: str

    def __repr__(self):
        return f"ValueReference({self.stage_id}, {self.unique_key})"


class RefcountedFuture:
    future: torch.futures.Future
    refcount: int

    def __init__(self, future, refcount):
        self.future, self.refcount = future, refcount

    def release(self):
        """
        Decrement refcount by 1. Return True if this instance should be freed
        """
        assert (
            self.refcount != 0
        ), "Detected reference counting inconsistency. Please report a bug to PiPPy"
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

    def __init__(
        self,
        rank,
        all_stages,
        max_outstanding=None,
        pp_rank=None,
        _record_mem_dumps=False,
        checkpoint=False,
    ):
        logging.info(f"[{rank}] Instantiating RankWorker")
        self.rank = rank
        self.all_stages = all_stages
        self.rank = rank
        self.pp_rank = pp_rank
        self._record_mem_dumps = _record_mem_dumps
        self.checkpoint = checkpoint

        # Maximum outstanding micro-batches of the pipeline schedule
        self.max_outstanding = max_outstanding
        # Keeps track of the outstanding micro-batches in current rank executor
        self.outstanding = 0
        self.stage_executors: Dict[int, PipeStageExecutor] = {}
        self.events: List[Event] = []

        self.waiting_runlist_lock = threading.Lock()
        # self.waiting_runlist (*and the contained WorkItems*) are guarded by
        # self.waiting_runlist_lock
        self.waiting_runlist: Dict[str, WorkItem] = {}

        self.ready_runlist_lock = threading.Lock()
        self.ready_runlist_cv = threading.Condition(self.ready_runlist_lock)
        self.ready_runlist: Dict[str, WorkItem] = {}

        self.worker_thread = threading.Thread(
            target=self.worker_loop, name=f"worker_{self.rank}", daemon=True
        )
        self.worker_thread.start()

    def create_stage_executor(self, stage_id, mod, mod_name):
        if stage_id in self.stage_executors:
            raise AssertionError(
                f"Rank {self.rank} already has stage {stage_id}"
            )

        assert (
            mod is not None or mod_name is not None
        ), "PipeStageExecutor requires mod or mod_name"

        if mod is None:
            with Pipe.stage_init_cv:
                defer_called = Pipe.stage_init_cv.wait_for(
                    Pipe.is_stage_init_deferred,
                    timeout=100,  # stop waiting after 100s
                )
                if not defer_called:
                    raise AssertionError(
                        f"Rank {self.rank} did not defer stage {stage_id} initialization "
                        f"though pipeline driver expect it to do so."
                    )

        self.stage_executors[stage_id] = PipeStageExecutor(
            stage_id=stage_id,
            mod=mod or Pipe.materialize_stage(mod_name),  # type: ignore[attr-defined]
            rank_worker=self,
            _record_mem_dumps=self._record_mem_dumps,
        )
        return self.stage_executors[stage_id]

    def enqueue_ready_runlist(self, unique_key, work_item):
        with self.ready_runlist_cv:
            logging.debug(
                f"[{self.rank}] Current ready runlist keys: {self.ready_runlist.keys()}"
            )
            self.ready_runlist[unique_key] = work_item
            self.ready_runlist_cv.notify()

    def enqueue_waiting_runlist(self, unique_key, work_item):
        with self.waiting_runlist_lock:
            logging.debug(
                f"[{self.rank}] Current waiting runlist keys: {self.waiting_runlist.keys()}"
            )
            assert (
                unique_key not in self.waiting_runlist
            ), f"key {unique_key} already in waiting runlist {self.waiting_runlist}"
            self.waiting_runlist[unique_key] = work_item

    def worker_loop(self):
        batch_id_to_remaining_backward_microbatches: Dict[int, int] = {}
        while True:
            work_item = None
            with self.ready_runlist_cv:
                while len(self.ready_runlist) == 0:
                    self.ready_runlist_cv.wait()

                logging.debug(
                    f"[{self.rank}] Dequeueing workitem from set of {len(self.ready_runlist)}"
                )
                # TODO: extra priorities
                for key in iter(self.ready_runlist.keys()):
                    # Skip forward work items if we hit the max outstanding limit
                    # If there are no other READY WorkItems, the runloop wraps around to the beginning and blocks again,
                    # waiting for another scheduled WorkItem to wake it back up. This works because the only condition
                    # that can schedule a WAITING Workitem is if another backward WorkItem executes and reduces the number
                    # of outstanding mciro-batches;
                    # If there are other READY WorkItems, the runloop executes as normally processing those
                    if (
                        self.ready_runlist[key].phase == Phase.FORWARD
                        and self.max_outstanding is not None
                        and self.outstanding >= self.max_outstanding
                    ):
                        continue
                    work_item = self.ready_runlist.pop(key)
                    break

            # We may not fetch any actionable work item in the above loop, go
            # back to the loop in this case
            if work_item is None:
                continue
            logging.debug(
                f"[{self.rank}][{work_item.microbatch_id}] Got WorkItem {work_item}"
            )

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
                raise RuntimeError(
                    f"Rank {self.rank} does not have stage {work_item.stage_id}"
                    f"Current keys {self.stage_executors.keys()}"
                )

            batch_id = work_item.batch_id
            num_microbatches = work_item.num_microbatches

            if batch_id not in batch_id_to_remaining_backward_microbatches:
                batch_id_to_remaining_backward_microbatches[
                    batch_id
                ] = num_microbatches

            start_ts = time.time()
            name = event_name(
                work_item.phase, work_item.stage_id, work_item.microbatch_id
            )
            id = event_id(
                work_item.phase,
                work_item.stage_id,
                work_item.microbatch_id,
                work_item.batch_id,
            )
            if self._record_mem_dumps:
                stage_executor._record_dumps_on_all_peer_executors(
                    f"M{id}_start", start_ts
                )

            value_ref_arg_idx = 0

            def retrieve_value_ref_args_by_idx(a):
                if isinstance(a, ValueReference) and a.unique_key != "noop":
                    nonlocal value_ref_arg_idx
                    val = ready_args[value_ref_arg_idx]
                    value_ref_arg_idx += 1
                    return val
                else:
                    return a

            args = pippy.fx.node.map_aggregate(
                args_value_refs, retrieve_value_ref_args_by_idx
            )
            kwargs = pippy.fx.node.map_aggregate(
                kwargs_value_refs, retrieve_value_ref_args_by_idx
            )

            def forward(args, kwargs, no_grad):
                args, flat_args = flatten_args_detach(args)
                kwargs, flat_kwargs = flatten_args_detach(kwargs)
                # Contains all tensors from args and kwargs, in flattened form
                flat_args += flat_kwargs

                logging.info(
                    f"[{self.rank}] Running forward module for microbatch {work_item.microbatch_id}"  # type: ignore[union-attr]
                )

                def forward_maybe_with_ddp(args, kwargs):
                    if isinstance(
                        stage_executor.mod,
                        torch.nn.parallel.distributed.DistributedDataParallel,
                    ):
                        with stage_executor.mod.no_sync():  # type: ignore[operator]
                            out_val = stage_executor.mod(*args, **kwargs)
                    else:
                        out_val = stage_executor.mod(*args, **kwargs)
                    return out_val

                def set_requires_grad(a):
                    if isinstance(a, torch.Tensor) and a.is_floating_point():
                        a.requires_grad_(True)
                    return a

                def dont_traverse_size(a):
                    return type(a) != torch.Size

                if no_grad:
                    with torch.no_grad():
                        out_val = forward_maybe_with_ddp(args, kwargs)
                        out_val = pippy.fx.node.map_aggregate(
                            out_val, set_requires_grad, dont_traverse_size
                        )
                else:
                    with torch.enable_grad():
                        out_val = forward_maybe_with_ddp(args, kwargs)

                return out_val, flat_args

            if phase == Phase.BACKWARD:
                if self.checkpoint:
                    logging.debug(
                        f"[{self.rank}][{work_item.microbatch_id}] Running backward phase. "
                        f"Rerunning forward because of checkpointing"
                    )
                    f_args, f_kwargs = stage_executor.fwd_cache.pop(
                        microbatch_id
                    )
                    out_val, flat_tensor_args = forward(
                        f_args, f_kwargs, no_grad=False
                    )
                    kwargs = dict(kwargs)
                    kwargs["stage_output"], kwargs["input_values"] = (
                        out_val if isinstance(out_val, tuple) else (out_val,),
                        flat_tensor_args,
                    )
                else:
                    logging.debug(
                        f"[{self.rank}][{work_item.microbatch_id}] Running backward phase. "
                        f"Retrieving stashed values"
                    )
                    # HACK: here we are directly accessing the saved tensor outputs
                    # for closed-over outputs so that they still have the grad_fn
                    # from local autograd. Can we solve this more elegantly?
                    kwargs = dict(kwargs)
                    (
                        kwargs["stage_output"],
                        kwargs["input_values"],
                    ) = stage_executor.fwd_cache.pop(microbatch_id)

            if work_item.phase == Phase.FORWARD:
                self.outstanding += 1
                out_val, flat_tensor_args = forward(
                    args, kwargs, no_grad=self.checkpoint
                )
                if self.checkpoint:
                    stage_executor.fwd_cache[microbatch_id] = args, kwargs
                else:
                    stage_executor.fwd_cache[microbatch_id] = (
                        out_val if isinstance(out_val, tuple) else (out_val,),
                        flat_tensor_args,
                    )

            elif work_item.phase == Phase.BACKWARD:
                logging.info(
                    f"[{self.rank}] Running backward for microbatch {work_item.microbatch_id}"
                )

                batch_id_to_remaining_backward_microbatches[batch_id] -= 1

                if (
                    isinstance(
                        stage_executor.mod,
                        torch.nn.parallel.distributed.DistributedDataParallel,
                    )
                    and batch_id_to_remaining_backward_microbatches[batch_id]
                    == 0
                ):
                    # HACK: reaching into DDP implementation details here. Is there a better way?
                    stage_executor.mod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                        list(
                            torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                                kwargs["stage_output"]
                            )
                        )
                    )

                out_val = stage_backward(*args, **kwargs)

                # Schedule forward stage of a new micro-batch
                self.outstanding -= 1
            elif work_item.phase == Phase.ACCUMULATE_GRAD:
                logging.debug(
                    f"[{self.rank}][{work_item.microbatch_id}] Running accumulate grad"
                )
                out_val = _null_coalesce_accumulate(*args, **kwargs)
            elif work_item.phase == Phase.SYNC_BARRIER:
                logging.debug(
                    f"[{self.rank}][{work_item.microbatch_id}] Running sync_barrier"
                )
                out_val = sync_barrier(*args, **kwargs)
            else:
                assert (
                    False
                ), f"Unrecognized phase {work_item.phase} encountered in execution"

            logging.debug(
                f"[{self.rank}][{work_item.microbatch_id}] Populating result of type {type(out_val)} "
                f"for {key}"
            )
            future.set_result(out_val)
            work_item.state = SchedState.DONE

            prev_name = prev_event_name(
                work_item.phase,
                self.all_stages,
                work_item.stage_id,
                work_item.microbatch_id,
            )
            next_name = next_event_name(
                work_item.phase,
                self.all_stages,
                work_item.stage_id,
                work_item.microbatch_id,
            )
            finish_ts = time.time()
            self.record_event(
                rank=self.rank,
                start_ts=start_ts,
                finish_ts=finish_ts,
                id=id,
                name=name,
                type=work_item.phase,
                mbid=work_item.microbatch_id,
            )
            self.record_event_dependency(
                from_id=prev_name, to_id=name, type="transfer"
            )
            self.record_event_dependency(
                from_id=name, to_id=next_name, type="transfer"
            )

            if self._record_mem_dumps:
                stage_executor._record_dumps_on_all_peer_executors(
                    f"M{id}_finish", finish_ts
                )

    # For work item marked with runlist_key, update its operand list with value
    def update_run_list(self, runlist_key, arg_idx, value):
        with self.waiting_runlist_lock:
            work_item = self.waiting_runlist[runlist_key]
            work_item.ready_args[arg_idx] = value
            work_item.blocked_args_count -= 1
            if work_item.blocked_args_count == 0:
                with self.ready_runlist_cv:
                    work_item.state = SchedState.READY
                    self.ready_runlist[runlist_key] = self.waiting_runlist.pop(
                        runlist_key
                    )
                    self.ready_runlist_cv.notify()
                logging.debug(
                    f"[{self.rank}][{work_item.microbatch_id}] all operands ready: {runlist_key}"
                )


class PipeStageExecutor(EventRecorder):
    """
    PipeStageExecutor encapsulates the execution semantics of a fragment of
    code on a pipeline stage. PipeStageExecutor handles:

    * Ownership of the stage's module and its recursive submodules/parameters
    * Serving as an entrypoint for the driver to push jobs into RankWorker's queue
    * TODO: gradient checkpointing
    """

    def __init__(self, stage_id, mod, rank_worker, _record_mem_dumps=False):
        logging.info(f"Instantiating PipeStageExecutor for stage {stage_id}")
        self.stage_id = stage_id
        self.mod = mod
        self.rank_worker = rank_worker
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}

        self.value_store_lock = threading.Lock()
        self.value_store_cv = threading.Condition(self.value_store_lock)
        self.value_store: Dict[str, RefcountedFuture] = {}

        self.peer_executors: Dict[int, torch._C._distributed_rpc.PyRRef] = None  # type: ignore[assignment]
        self._record_mem_dumps = _record_mem_dumps

        self.optimizer = None
        # Used to ensure optimizer is created before we create learning rate scheduler
        self.optim_init_lock = threading.Lock()
        self.optim_init_cv = threading.Condition(self.optim_init_lock)

        self.lr_scheduler = None
        self.device = self._find_mod_device()

        # Send/recv order normalization
        self.callee_send_tag: Dict[int, int] = {}  # callee stage: tag seq num
        self.caller_recv_tag: Dict[int, int] = {}  # caller stage: tag seq num
        self.callee_send_tag_lock = threading.Lock()
        self.caller_recv_tag_lock = threading.Lock()
        self.caller_recv_tag_cv = threading.Condition(self.caller_recv_tag_lock)

    def _find_mod_device(self):
        # We assume that all parameters in the module are on the same device
        # HACK: we assume the module has at least one parameter
        param = next(self.mod.parameters(), None)
        buffer = next(self.mod.buffers(), None)
        if param is not None:
            device = param.device
        elif buffer is not None:
            device = buffer.device
        else:
            logging.warning(
                f"Module of stage {self.stage_id} has no parameter or buffer, "
                f"cannot figure out device. Setting it to cpu"
            )
            device = torch.device("cpu")
        return device

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

    def init_data_parallel(self, n_stages, dp_group_size, dp_pg_cb=None):
        worker_rank = self.rank_worker.rank
        if dp_pg_cb is not None:
            logging.info(
                f"Rank[{worker_rank}] stage[{self.stage_id}] Initializing data parallel: "
                f"using DP process groups provided by user"
            )
            self.mod = torch.nn.parallel.DistributedDataParallel(
                self.mod, process_group=dp_pg_cb(self.stage_id)
            )
            return

        logging.debug(
            f"Rank[{worker_rank}] stage[{self.stage_id}] Initializing data parallel: "
            f"creating DP process groups internally"
        )
        # Discover DP peers via Store
        # HACK: using the Store coming with the default process group
        _store = torch.distributed.distributed_c10d._get_default_store()
        # Wrap default store by adding a prefix to each key inserted so as not to step into default store's space
        store = torch.distributed.PrefixStore("PiPPy", _store)
        # TODO: figure out the unique global "stage rank" for Interleaved 1F1B
        my_rank = str(worker_rank)
        my_stage = str(self.stage_id)
        # Each stage rank checks in with their stage id in respective pipe
        store.set(my_rank, my_stage)

        # Create a mapping from stage id to DP ranks
        stage_to_dp_ranks: Dict[int, List[int]] = {}
        for stage in range(n_stages):
            stage_to_dp_ranks.setdefault(stage, [])

        # Wait for all stages to check in
        world_size = n_stages * dp_group_size
        all_ranks = [str(i) for i in range(world_size)]
        store.wait(all_ranks)
        logging.debug(
            f"Rank[{worker_rank}] stage[{self.stage_id}] Initializing data parallel: all stages have checked in"
        )

        # Fill the mapping
        for rank in all_ranks:
            stage = store.get(rank)  # type: ignore[assignment]
            stage_to_dp_ranks[int(stage)].append(int(rank))

        # Create DP process group for each stage
        # Note: even if a rank is not in the DP group of another stage, it must still participate in the new_group call of
        # that stage; this is required by c10d
        for stage in range(n_stages):
            dp_group_ranks = stage_to_dp_ranks[stage]
            dp_pg_for_stage = torch.distributed.new_group(dp_group_ranks)
            if stage == self.stage_id:
                logging.info(
                    f"Rank[{worker_rank}] stage[{self.stage_id}] "
                    f"DP group {dp_group_ranks} -- init complete"
                )

            # Wrap stage module with DDP using the DP group corresponding to own stage
            if self.stage_id == stage:
                self.mod = torch.nn.parallel.DistributedDataParallel(
                    self.mod, process_group=dp_pg_for_stage
                )

    def create_future(self):
        # Future constructor does not accept CPU device, must set to None
        return torch.futures.Future(
            devices=None if self.device.type == "cpu" else [self.device]
        )

    def invoke(
        self,
        output_unique_key: str,
        phase: Phase,
        args,
        kwargs,
        cur_microbatch: int,
        debug_str: str,
        output_refcount: int,
        batch_id: int,
        num_microbatches: int,
    ):
        start_ts = time.time()
        target_name = event_name(phase, self.stage_id, cur_microbatch)
        target_id = event_id(phase, self.stage_id, cur_microbatch, batch_id)
        name = f"R{target_name}"
        id = f"R{target_id}"
        if self._record_mem_dumps:
            self._record_dumps_on_all_peer_executors(f"M{id}_invoke", start_ts)
        # TODO: do we need to serialize calls to invoke() to preserve the order in which WorkItems appear for
        # static schedules?

        logging.debug(
            f"[{self.stage_id}][{cur_microbatch}] Received invoke call for {debug_str}"
        )
        # Extract all ValueRef arguments so we can spawn asynchronous data transfers
        # for each of them
        value_ref_args: List[ValueReference] = []

        def extract_value_ref_args(arg):
            if isinstance(arg, ValueReference) and arg.unique_key != "noop":
                value_ref_args.append(arg)

        pippy.fx.node.map_aggregate(args, extract_value_ref_args)
        pippy.fx.node.map_aggregate(kwargs, extract_value_ref_args)

        logging.debug(
            f"[{self.stage_id}][{cur_microbatch}] Invoke call found {len(value_ref_args)} ValueReference arguments"
        )

        # Construct WorkItem for this microbatch+phase and record it in the
        # waiting runlist

        # We provide device to the Future constructor so that between
        # future.set_result() and future.wait() correct dependencies can be
        # captured
        # We assume the output value is on the same device as the stage's parameters

        # Future constructor does not accept CPU device, must set to None
        future: torch.futures.Future = self.create_future()

        # TODO: increase blocked_args_count for extra things like scheduling
        work_item = WorkItem(
            stage_id=self.stage_id,
            phase=phase,
            args=args,
            kwargs=kwargs,
            future=future,
            microbatch_id=cur_microbatch,
            blocked_args_count=len(value_ref_args),
            ready_args={},
            batch_id=batch_id,
            num_microbatches=num_microbatches,
            debug_str=debug_str,
        )
        logging.debug(
            f"[{self.stage_id}][{cur_microbatch}] Invoke instantiated WorkItem {work_item} with key {output_unique_key}"
        )
        if len(value_ref_args) == 0:
            # TODO: convert initial input into ValueRef?
            # We always put this work item into the ready queue, though we mark
            # it with different state flags depending on whether the schedule
            # would hold it based on max outstanding allowed
            work_item.state = SchedState.READY
            logging.debug(
                f"[{self.stage_id}][{cur_microbatch}] No RRef arguments. "
                f"Scheduling directly as READY workitem"
            )
            self.rank_worker.enqueue_ready_runlist(output_unique_key, work_item)
        else:
            logging.debug(
                f"[{self.stage_id}][{cur_microbatch}] Scheduling WorkItem as WAITING workitem"
            )
            work_item.state = SchedState.WAITING
            self.rank_worker.enqueue_waiting_runlist(
                output_unique_key, work_item
            )

        # Group Value Ref Args based on source stage
        # `callee_stage_dict` has the following structure:
        #   Dict[callee_stage, Dict[my_arg_idx, value_ref]]
        callee_stage_dict: Dict[int, Dict[int, ValueReference]] = {}
        for arg_idx, value_ref_arg in enumerate(value_ref_args):
            # Check if the ValRef corresponds to a tensor
            if "tensor_meta" in value_ref_arg.meta:
                callee_stage = value_ref_arg.stage_id
                batch_refs = callee_stage_dict.setdefault(callee_stage, {})
                batch_refs[arg_idx] = value_ref_arg
            else:
                # For non-tensor (e.g. a value or a size vector), we use RPC to spawn asynchronous data transfer
                logging.debug(
                    f"[{self.stage_id}][{cur_microbatch}] Launching RPC data transfer for "
                    f"ValueReference {arg_idx} {value_ref_arg}"
                )
                self.async_transfer(
                    cur_microbatch, value_ref_arg, arg_idx, output_unique_key
                )

        # For tensors, we use c10d two-sided send/recv
        # Batch call per source stage to reduce number of RPC threads
        with self.callee_send_tag_lock:
            for callee_stage, batch_refs in callee_stage_dict.items():
                value_ref_executor_rref = self.peer_executors[callee_stage]
                tag = self.callee_send_tag.setdefault(callee_stage, 0)
                self.callee_send_tag[callee_stage] += 1
                value_ref_executor_rref.rpc_async().batch_send(
                    self.stage_id,
                    output_unique_key,
                    cur_microbatch,
                    batch_refs,
                    tag,
                )
                self.batch_recv(
                    cur_microbatch,
                    output_unique_key,
                    callee_stage,
                    batch_refs,
                    tag,
                )

        with self.value_store_cv:
            assert output_unique_key not in self.value_store, (
                f"[{self.stage_id}] Output key {output_unique_key} "
                f"already exists or is not consumed from previous batch"
            )
            self.value_store[output_unique_key] = RefcountedFuture(
                future, output_refcount
            )
            self.value_store_cv.notify_all()

        finish_ts = time.time()
        self.record_event(
            rank=self.rank_worker.rank,
            start_ts=start_ts,
            finish_ts=finish_ts,
            id=id,
            name=name,
            type="received",
            mbid=cur_microbatch,
        )
        self.record_event_dependency(
            from_id=name, to_id=target_name, type="waiting"
        )

        return ValueReference(self.stage_id, output_unique_key)

    def coalesced_index_value(
        self, indices: List[Tuple[str, int, ValueReference, int]]
    ):
        for index_tuple in indices:
            # `output_unique_key` is the key for the indexed output value (single)
            # `value_ref.unique_key` is the key for the overall output of current stage (can have multiple values)
            (output_unique_key, output_refcount, value_ref, idx) = index_tuple
            logging.debug(
                f"[{self.stage_id}] Received getitem call: {(output_unique_key, output_refcount, value_ref, idx)}"
            )
            with self.value_store_cv:
                # TODO: investigate why value reference in the last batch has not been fully consumed
                if output_unique_key in self.value_store:
                    logging.debug(
                        f"[{self.stage_id}] Indexed value already in store: {(output_unique_key, output_refcount, value_ref, idx)}"
                    )
                    # raise RuntimeError(f'Repeated index value call detected, potentially due to getitem calls not consumed in previous batch')

                # Wait for the future representing the stage output to be created
                while value_ref.unique_key not in self.value_store:
                    self.value_store_cv.wait()
                # Now the stage output future is created
                refcounted_future = self.value_store[value_ref.unique_key]

                # For the purposes of refcounting, decrement this use
                if refcounted_future.release():
                    self.value_store.pop(value_ref.unique_key)

                # Create an indexed future that represents a specific output arg
                # Here we use an attach functon so that the index passed to the lambda changes in every loop
                def attach_index(fut, index):
                    indexed_fut = fut.then(lambda f: f.value()[index])
                    return indexed_fut

                indexed = attach_index(refcounted_future.future, idx)

                # Enqueue the indexed future
                # And notify places that may be waiting for it to be created, such as get_value
                self.value_store[output_unique_key] = RefcountedFuture(
                    indexed, output_refcount
                )
                self.value_store_cv.notify_all()

    def get_value(
        self,
        caller_stage,
        runlist_key,
        microbatch,
        value_ref_arg,
    ):
        callee_stage = value_ref_arg.stage_id
        logging.debug(
            f"[{callee_stage}][{microbatch}] Executing transfer of value "
            f"{value_ref_arg} initiated by stage {caller_stage} for {runlist_key}"
        )
        assert (
            callee_stage == self.stage_id
        ), "Mismatch between ValueRef and stage executor"

        with self.value_store_cv:
            # Waiting for the indexed future for this arg to be created
            while value_ref_arg.unique_key not in self.value_store:
                self.value_store_cv.wait()
            # Now the indexed future is created
            refcounted_future = self.value_store[value_ref_arg.unique_key]

        value = refcounted_future.future.wait()

        with self.value_store_lock:
            if refcounted_future.release():
                self.value_store.pop(value_ref_arg.unique_key)

        return value

    def async_transfer(self, microbatch, value_ref_arg, arg_idx, runlist_key):
        logging.debug(
            f"[{self.stage_id}][{microbatch}] Requesting transfer of value {value_ref_arg} "
            f"for runlist item {runlist_key} arg_idx {arg_idx}"
        )
        callee_stage = value_ref_arg.stage_id
        value_ref_executor_rref = self.peer_executors[callee_stage]

        fut = value_ref_executor_rref.rpc_async().get_value(
            self.stage_id,
            runlist_key,
            microbatch,
            value_ref_arg,
        )

        def bottom_half(fut):
            logging.debug(
                f"[{self.stage_id}][{microbatch}] Completing transfer of value {value_ref_arg} "
                f"for runlist item {runlist_key} arg_idx {arg_idx}"
            )
            value = fut.value()
            self.rank_worker.update_run_list(runlist_key, arg_idx, value)

        return fut.then(bottom_half)

    def batch_send(
        self,
        caller_stage,
        runlist_key,
        microbatch,
        batch_refs,
        tag,
    ):
        # Wait till this batch's turn to send
        with self.caller_recv_tag_cv:
            self.caller_recv_tag.setdefault(caller_stage, 0)
            while self.caller_recv_tag[caller_stage] < tag:
                self.caller_recv_tag_cv.wait()

        logging.debug(
            f"[{self.stage_id}][{microbatch}] Sending batch {tag} of "
            f"{len(batch_refs)} values initiated by stage {caller_stage} for {runlist_key}"
        )

        for _, value_ref_arg in batch_refs.items():
            with self.value_store_cv:
                # Waiting for the indexed future for this arg to be created
                while value_ref_arg.unique_key not in self.value_store:
                    self.value_store_cv.wait()
                # Now the indexed future is created
                refcounted_future = self.value_store[value_ref_arg.unique_key]

            value = refcounted_future.future.wait()

            with self.value_store_lock:
                if refcounted_future.release():
                    self.value_store.pop(value_ref_arg.unique_key)

            # Instead of return value let's do a send call
            if torch.distributed.get_backend() == "gloo":
                # Gloo P2P does not support work.get_future, so we use send instead
                torch.distributed.send(value, caller_stage, tag=tag)
            else:
                torch.distributed.isend(value, caller_stage, tag=tag)

        # Notify next send that's potentially waiting
        with self.caller_recv_tag_cv:
            self.caller_recv_tag[caller_stage] += 1
            self.caller_recv_tag_cv.notify_all()

    def batch_recv(
        self, microbatch, runlist_key, callee_stage, batch_refs, tag
    ):
        logging.debug(
            f"[{self.stage_id}][{microbatch}] Receiving batch {tag} of {len(batch_refs)} values "
            f"for runlist item {runlist_key} from stage {callee_stage}"
        )
        futures = []

        for arg_idx, value_ref_arg in batch_refs.items():
            tm = value_ref_arg.meta["tensor_meta"]
            recv_buff = torch.empty(
                tm.shape, dtype=tm.dtype, device=self.device
            )

            if torch.distributed.get_backend() == "gloo":
                # Gloo P2P does not support work.get_future, so we need to:
                # - manually create the Future,
                # - use recv instead, and
                # - manually set_result to the Future
                fut: torch.futures.Future = self.create_future()
                torch.distributed.recv(recv_buff, callee_stage, tag=tag)
                fut.set_result(recv_buff)
            else:
                work = torch.distributed.irecv(recv_buff, callee_stage, tag=tag)
                fut = work.get_future()  # type: ignore[attr-defined]

            def bottom_half(fut):
                logging.debug(
                    f"[{self.stage_id}][{microbatch}] Completing transfer of value {value_ref_arg} "
                    f"for runlist item {runlist_key} arg_idx {arg_idx}"
                )
                value = fut.value()
                # It is awkward that the Work class in PyTorch fixes the result return to a List:
                #   def result(self) -> List[Tensor]: ...
                # See torch/_C/_distributed_c10d.pyi
                # We don't expect P2P operations to actually result in a List, hence unpacking and getting the first and
                # only tensor out
                if isinstance(value, List):
                    value = value[0]
                self.rank_worker.update_run_list(runlist_key, arg_idx, value)

            futures.append(fut.then(bottom_half))

        return futures

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
        with self.optim_init_cv:
            self.optimizer = optim_class(self.mod.parameters(), *args, **kwargs)
            self.optim_init_cv.notify()
        return self.optimizer

    def instantiate_lr_scheduler(self, lr_sched_class, *args, **kwargs):
        # Make sure optimizer has been created
        with self.optim_init_cv:
            while self.optimizer is None:
                self.optim_init_cv.wait()

        logging.info(f"[{self.stage_id}] Creating learning rate scheduler")
        self.lr_scheduler = lr_sched_class(self.optimizer, *args, **kwargs)
        return self.lr_scheduler

    def step_lr_scheduler(self, *args, **kwargs):
        self.lr_scheduler.step(*args, **kwargs)  # type: ignore[union-attr]

    def _check_cleanup(self) -> bool:
        if len(self.value_store):
            logging.warning(
                f"[{self.stage_id}] Unclean value store: {self.value_store}"
            )
            return False
        return True

    def _record_dump(self, dump_id, ts):
        first_param = next(self.mod.parameters(), None)
        device: torch.device = (
            first_param.device
            if first_param is not None
            else torch.device("cpu")
        )
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated()
            max_alloc = torch.cuda.max_memory_allocated()
            rsrvd = torch.cuda.memory_reserved()
            max_rsrvd = torch.cuda.max_memory_reserved()
            assert (
                alloc <= max_alloc
            ), f"alloc = {alloc} max_alloc = {max_alloc}"
            assert (
                rsrvd <= max_rsrvd
            ), f"rsrvd = {rsrvd} max_rsrvd = {max_rsrvd}"
            assert (
                max_alloc <= max_rsrvd
            ), f"max_alloc = {max_alloc} max_rsrvd = {max_rsrvd}"
            self.record_dump(
                rank=self.rank_worker.rank,
                ts=ts,
                id=dump_id,
                name=dump_id,
                type="dump",
                allocators={
                    "cuda.4.alloc": Allocator(
                        f"alloc_{self.rank_worker.rank}",
                        {
                            "size": alloc,
                        },
                    ),
                    "cuda.3.max_alloc-alloc": Allocator(
                        f"max_alloc-alloc_{self.rank_worker.rank}",
                        {
                            "size": max_alloc - alloc,
                        },
                    ),
                    "cuda.2.rsrvd-max_alloc": Allocator(
                        f"rsrvd-max_alloc_{self.rank_worker.rank}",
                        {
                            "size": max(rsrvd - max_alloc, 0),
                        },
                    ),
                    "cuda.1.max_rsrvd-max_alloc_or_rsrvd": Allocator(
                        f"max_rsrvd-max_alloc_or_rsrvd_{self.rank_worker.rank}",
                        {
                            "size": max_rsrvd
                            - (max_alloc if max_alloc > rsrvd else rsrvd),
                        },
                    ),
                },
            )

    def _record_dumps_on_all_peer_executors(self, id, ts):
        for peer_executor_rref in self.peer_executors.values():
            peer_executor_rref.rpc_sync()._record_dump(f"{id}", ts)


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


class PipelineOptimizer(torch.optim.Optimizer):
    def __init__(self, remote_optims):
        self.remote_optims = remote_optims

        # TODO: enable this
        # self._hook_for_profile()

        # TODO: enable this
        # self.state = defaultdict(dict)

        self.param_groups = []

        # Collect RRefs to remote parameters
        param_group = {"params": []}  # type: ignore[var-annotated]

        for optim in self.remote_optims:
            remote_state = optim.rpc_sync().__getstate__()
            assert isinstance(remote_state, dict)
            for group in remote_state["param_groups"]:
                param_group["params"].extend(group["params"])
                for k in group:
                    if k != "params":
                        param_group.setdefault(k, group[k])

        self.param_groups = [param_group]

    def __getstate__(self):
        raise NotImplementedError()

    def __setstate__(self, state):
        raise NotImplementedError()

    def _hook_for_profile(self):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state_dict):
        raise NotImplementedError()

    # PyTorch type annotation for this function is wrong. See
    # https://github.com/pytorch/pytorch/pull/76998 for proposed fix
    def zero_grad(self, set_to_none: bool = False):  # type: ignore
        futs = []
        for optim in self.remote_optims:
            futs.append(optim.rpc_async().zero_grad(set_to_none))
        _wait_for_all(futs)

    def step(self, closure=None):
        futs = []
        for optim in self.remote_optims:
            futs.append(optim.rpc_async().step(closure))
        _wait_for_all(futs)

    def add_param_group(self, param_group):
        raise NotImplementedError()


class PipelineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, stage_to_scheds, stage_to_executor):
        # A dict from stage id to LR schedulers
        self.stage_to_scheds = stage_to_scheds
        self.stage_to_executor = stage_to_executor
        self.new_step_called = False
        self.last_lr = []

    def step(self, *args, **kwargs):
        futs = []
        # Step all remote LR schedulers

        # We use the executor block below because calling scheduler.step()
        # remotely might cause pickling nested functions, where these nested
        # functions are usually defined inside user's lr scheduler constructor
        # as lambda functions to be used by the lr scheduler
        # See https://github.com/pytorch/PiPPy/issues/404
        """
        for scheduler in self.stage_to_scheds.values():
            futs.append(scheduler.rpc_async().step(*args, **kwargs))
        """
        for executor in self.stage_to_executor.values():
            futs.append(executor.rpc_async().step_lr_scheduler(*args, **kwargs))

        _wait_for_all(futs)
        # Mark new step (invalidates last_lr)
        self.new_step_called = True

    def get_last_lr(self):
        """Return last computed learning rate by remote schedulers."""
        # No need to involve remote schedulers if no new step calls
        if not self.new_step_called:
            return self.last_lr

        # Ask LR scheduler of stage 0 to return new learning rate as representation of all stages, because:
        # (i) we do not support multiple parameter groups yet (neither PipelineOptimizer nor PipelineLRScheduler does),
        # so there are not param group specific LR's; and
        # (ii) current LRS implementations do not relies on state within the optimizer, so the LR's of different stages
        # will not diverge
        assert self.stage_to_scheds, "No learning rate scheduler"
        self.last_lr = self.stage_to_scheds[0].remote().get_last_lr().to_here()
        self.new_step_called = False
        return self.last_lr

    def state_dict(self):
        """Returns the state of the remote schedulers as a :class:`dict`"""
        # Ask LR scheduler of stage 0 to return state_dict as representation of all stages, for the same reason as
        # stated in get_last_lr()
        rv: Dict = {}
        assert self.stage_to_scheds, "No learning rate scheduler"
        rv = self.stage_to_scheds[0].remote().state_dict().to_here()
        return rv

    def load_state_dict(self, state_dict):
        """Loads the scheduler state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        futs = []
        for scheduler in self.stage_to_scheds.values():
            futs.append(scheduler.rpc_async().load_state_dict(state_dict))

        _wait_for_all(futs)

    def get_lr(self):
        # Even in single scheduler setting, get_lr is more of an internal method to be called by step()
        # See: pytorch/torch/optim/lr_scheduler.py
        warnings.warn(
            "To get the last learning rate computed by the scheduler, "
            "please use `get_last_lr()`."
        )
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate."""
        # This is more of an internal method of native scheduler
        # See: pytorch/torch/optim/lr_scheduler.py
        raise NotImplementedError


class PipelineDriverBase(torch.nn.Module):
    def __init__(
        self,
        pipe: Pipe,
        chunks: int,
        world_size: int,
        all_ranks: List[int] = None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
        _debug_mask_minibatches: bool = False,
        max_outstanding=None,
        interleave_stages=False,
        _record_mem_dumps=False,
        checkpoint=False,
        use_c10d=False,
        loss_reducer: LossReducer = sum_reducer,
    ):
        super().__init__()
        self.pipe = pipe
        self.chunks = chunks
        self.world_size = world_size
        self.all_ranks = all_ranks
        self.args_chunk_spec = args_chunk_spec
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.loss_reducer = loss_reducer
        self.output_chunk_spec = (
            output_chunk_spec
            if output_chunk_spec
            else gen_output_chunk_spec(pipe.loss_spec, loss_reducer)
        )

        # Maximum outstanding micro-batches allowed by the pipeline schedule
        # None means no limit
        self.max_outstanding: Optional[int] = max_outstanding
        self._debug_mask_minibatches = _debug_mask_minibatches
        self.interleave_stages = interleave_stages

        self.microbatch_interpreters: List[RemoteInterpreter] = []
        self.batch_id = 0
        self._record_mem_dumps = _record_mem_dumps
        self.optimizer_inited = False
        self.checkpoint = checkpoint
        self.use_c10d = use_c10d

    def _init_remote_executors(self):
        self.rank_worker_rrefs: Dict[int, torch.distributed.rpc.RRef] = {}
        self.remote_stage_executor_rrefs: Dict[  # type: ignore[syntax]
            str, (int, torch.distributed.rpc.RRef)
        ] = {}

        if self.all_ranks is not None:
            assert (
                len(self.all_ranks) == self.world_size
            ), "Explicitly specified ranks must match world_size"
        else:
            self.all_ranks = list(range(self.world_size))
        logging.info(
            f"[root] Creating pipeline driver with {self.world_size} workers: {self.all_ranks}"
        )

        class ExecutorDescriptor:
            name: str
            mod: Optional[torch.nn.Module]
            has_backward: bool = False

        split_gm = self.pipe.split_gm

        executor_descriptors = []
        bw_idx = -1
        for node in split_gm.graph.nodes:
            if node.op == "call_module":
                descr = ExecutorDescriptor()
                descr.name = node.target
                if Pipe.is_stage_init_deferred():
                    descr.mod = None
                else:
                    descr.mod = split_gm.get_submodule(node.target)
                executor_descriptors.append(descr)
            elif (node.op, node.target) == ("call_function", stage_backward):
                executor_descriptors[bw_idx].has_backward = True
                node.meta["fw_stage"] = executor_descriptors[bw_idx].name
                bw_idx -= 1
            elif (node.op, node.target) == (
                "call_function",
                _null_coalesce_accumulate,
            ):
                node.meta["fw_stage"] = executor_descriptors[bw_idx].name

        assert all(d.has_backward for d in executor_descriptors) or all(
            not d.has_backward for d in executor_descriptors
        )

        if len(executor_descriptors) > self.world_size:
            if not self.interleave_stages:
                raise RuntimeError(
                    f"Tried to run pipeline with {len(executor_descriptors)} stages with a world size of "
                    f"{self.world_size}. Please ensure world_size is large enough to accommodate your pipeline."
                )

        ranks_to_launch = self.world_size
        n_stages = len(executor_descriptors)
        if n_stages < self.world_size:
            ranks_to_launch = n_stages
            warnings.warn(
                f"Running pipeline with {n_stages} stages on world_size of {self.world_size}. "
                f"Remaining ranks will be idle."
            )

        if self.interleave_stages and n_stages <= ranks_to_launch:
            self.interleave_stages = False
            warnings.warn(
                "Falling back from Interleaved 1F1B to 1F1B "
                "since there are enough ranks to support one stage per rank"
            )

        # Fire up rank workers
        all_stages = list(range(n_stages))
        pp_rank = 0
        for rank in self.all_ranks[:ranks_to_launch]:
            kwargs = {
                "rank": rank,
                "all_stages": all_stages,
                "max_outstanding": self.max_outstanding,
                "pp_rank": pp_rank,
                "_record_mem_dumps": self._record_mem_dumps,
                "checkpoint": self.checkpoint,
            }
            self.rank_worker_rrefs[rank] = rpc.remote(
                rank, RankWorker, args=(), kwargs=kwargs
            )
            pp_rank += 1

        self.stage_to_executor: Dict = {}

        # Ask each RankWorker to create stage thereon
        # This can involve checkpoint loading in deferred init case
        for stage_id, descr in enumerate(executor_descriptors):
            # Assign stages to rank workers in a round-robin fashion
            rank = self.all_ranks[stage_id % self.world_size]
            logging.debug(f"[root] Sending stage_id = {stage_id} mod to worker")
            self.remote_stage_executor_rrefs[descr.name] = (
                stage_id,
                self.rank_worker_rrefs[rank]
                .remote()
                .create_stage_executor(
                    stage_id=stage_id,
                    mod=descr.mod,
                    mod_name=descr.name,
                ),
            )

        # Check that each RankWorker has completed stage init
        for stage_id, descr in enumerate(executor_descriptors):
            logging.debug(
                f"[root] Waiting stage_id = {stage_id} mod to be confirmed by worker"
            )
            while not self.remote_stage_executor_rrefs[descr.name][
                1
            ].confirmed_by_owner():
                pass

            self.stage_to_executor[stage_id] = self.remote_stage_executor_rrefs[
                descr.name
            ][1]

        # Inform executors of their peers
        for stage_id, executor in self.stage_to_executor.items():
            executor.rpc_sync().install_peer_executors(self.stage_to_executor)

    """
    Method for creating a data parallel clique for each stage, across multiple pipelines
        dp_group_size: size of each data parallel group, equals to the number of pipelines
        dp_pg_cb: optional Callable taking pipeline stage as argument and returning corresponding data parallel group;
                  user can use this Callable to pass in prepared data parallel groups
    """

    def init_data_parallel(self, dp_group_size, dp_pg_cb=None):
        if dp_group_size <= 1:
            logging.info(
                "[root] Data parallel group size <= 1, skipping data parallel initialization"
            )
            return

        n_stages = len(self.stage_to_executor)
        logging.info(
            f"[root] Initializing {n_stages} data parallel groups, each of size {dp_group_size}"
        )
        futs = []
        # Asks all stage executors to participate in DP process group init
        # These must be async calls because otherwise there will be deadlocks
        for executor in self.stage_to_executor.values():
            futs.append(
                executor.rpc_async().init_data_parallel(
                    n_stages, dp_group_size, dp_pg_cb
                )
            )

        # Here we wait for all DP process groups to be initialized before the user can ask the PipeDriver to run
        _wait_for_all(futs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "PipelineDriverBase is an abstract base class, please use a concrete "
            "implementation class."
        )

    def train(self, mode=True):
        for executor in self.stage_to_executor.values():
            executor.rpc_sync().train(mode=mode)

    def eval(self):
        self.train(mode=False)

    def instantiate_optimizer(self, optim_class, *args, **kwargs):
        remote_optims = []
        # Keeps track of stage to optimizer mapping
        self.stage_to_optim: Dict = {}
        for stage, executor in self.stage_to_executor.items():
            if executor.rpc_sync()._should_instantiate_optim():
                remote_optim = executor.remote().instantiate_optimizer(
                    optim_class, *args, **kwargs
                )
                remote_optims.append(remote_optim)
                self.stage_to_optim[stage] = remote_optim

        self.optimizer_inited = True
        return PipelineOptimizer(
            [optim for optim in remote_optims if optim is not None]
        )

    """
    Create learning rate scheduler for the optimizer of the pipeline.
    Note: this API cannot be called before instantiate_optimizer is called.
    """

    def instantiate_lr_scheduler(self, lr_sched_class, *args, **kwargs):
        if not self.optimizer_inited:
            raise RuntimeError(
                "[root] instantiate_optimizer must be called before instantiate_lr_scheduler"
            )

        stage_to_scheds: Dict = {}
        for stage, optim in self.stage_to_optim.items():
            if optim is not None:
                executor = self.stage_to_executor[stage]
                remote_lr_sched = executor.remote().instantiate_lr_scheduler(
                    lr_sched_class, *args, **kwargs
                )
                stage_to_scheds[stage] = remote_lr_sched

        return PipelineLRScheduler(stage_to_scheds, self.stage_to_executor)

    def _sync_replicated_params(self):
        logging.debug(
            f"[root] Synchronizing gradients for {len(self.pipe.replicated_params)} sets of replicated parameters"
        )
        for param_set in self.pipe.replicated_params:
            grad_values = []
            for module_name, param_qualname in param_set.items():
                assert module_name in self.remote_stage_executor_rrefs
                stage_id, module_rref = self.remote_stage_executor_rrefs[
                    module_name
                ]
                grad_value = module_rref.rpc_sync().get_grad(param_qualname)
                grad_values.append(grad_value)

            synced_value = torch.sum(torch.stack(grad_values), dim=0)

            for module_name, param_qualname in param_set.items():
                assert module_name in self.remote_stage_executor_rrefs
                stage_id, module_rref = self.remote_stage_executor_rrefs[
                    module_name
                ]
                module_rref.rpc_sync().set_grad(param_qualname, synced_value)

    def _retrieve_output_values(self, microbatch_interpreters, last_nodes):
        logging.debug(
            f"[root] Retrieving output values from {len(microbatch_interpreters)} chunks"
        )
        output_vals = []
        for interp, last_node in zip(microbatch_interpreters, last_nodes):
            interp.run_until(lambda n: False)
            output_vals.append(interp.env[last_node])

        # First kick of async transfers to retrieve ValueReference values
        def initiate_async_transfer(a):
            if isinstance(a, ValueReference):
                value_ref_executor_rref = self.stage_to_executor[a.stage_id]
                return value_ref_executor_rref.rpc_async().get_value(
                    "root", "collect", -1, a
                )
            else:
                return a

        output_vals = pippy.fx.node.map_aggregate(
            output_vals, initiate_async_transfer
        )

        # Then wait for futures to be ready
        return pippy.fx.node.map_aggregate(
            output_vals,
            lambda a: a.wait() if isinstance(a, torch._C.Future) else a,
        )

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

    def _check_stages_cleanup(self) -> bool:
        clean = True
        for executor in self.stage_to_executor.values():
            clean &= executor.rpc_sync()._check_cleanup()
        return clean


class RemoteInterpreter(pippy.fx.Interpreter, EventRecorder):
    def __init__(
        self,
        remote_stage_executor_rrefs,
        stage_to_executor,
        module,
        cur_microbatch: int,
        args,
        kwargs,
        batch_id: int,
        num_microbatches: int,
        garbage_collect_values=True,
    ):
        super().__init__(module, garbage_collect_values)
        self.remote_stage_executor_rrefs = remote_stage_executor_rrefs
        self.stage_to_executor = stage_to_executor
        self.cur_microbatch = cur_microbatch
        self.pc = 0
        self.node_list = list(self.module.graph.nodes)
        logging.debug(
            f"[root] RemoteInterpreter created with {len(self.node_list)} nodes"
        )

        # Process args/kwargs

        # TODO: replace this with GraphModule.signature() when it lands
        parameters = []
        for node in self.module.graph.nodes:
            if node.op != "placeholder":
                continue
            default = next(iter(node.args)) if node.args else Parameter.empty
            parameters.append(
                Parameter(
                    node.name, Parameter.POSITIONAL_OR_KEYWORD, default=default
                )
            )

        # We are building a safety net here in case user passes in extra arguments than those defined as variable
        # arguments (i.e. non-concrete args) at the tracing phase
        # TODO: Remove this safety net
        traced_args = [p.name for p in parameters]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in traced_args}
        if len(filtered_kwargs) != len(kwargs):
            extra_args = kwargs.keys() - filtered_kwargs.keys()
            warnings.warn(
                f"Received extra arguments: {extra_args}. "
                f"They might have already been given a concrete value during pipeline compilation via `concrete_args`. "
                f"We will ignore the current inputs and use the values given during compilation."
            )

        sig = Signature(parameters)
        bound_args = sig.bind(*args, **filtered_kwargs)
        bound_args.apply_defaults()
        self.args = bound_args.args
        self.args_iter = iter(self.args)
        self.batch_id = batch_id
        self.num_microbatches = num_microbatches
        # Dict from stage id to a list holding the coalesced getitem indices
        self.stage_output_indices: Dict[
            int, List[Tuple[str, int, ValueReference, int]]
        ] = {}

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        node = self.node_list[self.pc]
        # if PipelineDriver is running inside `torch.no_grad()` context manager then `stage_backward*` nodes
        # are excluded from execution, so we need exclude `stage_backward*` from reference count, otherwise
        # it will cause memory leak.
        users = (
            list(
                filter(
                    lambda user: not user.name.startswith("stage_backward"),
                    node.users.keys(),
                )
            )
            if not torch.is_grad_enabled()
            else node.users.keys()
        )
        if target in self.remote_stage_executor_rrefs:
            stage_id, stage_executor = self.remote_stage_executor_rrefs[target]
            logging.debug(
                f"[root][{self.cur_microbatch}] Issuing {Phase.FORWARD} "
                f"invocation for target {target} on stage {stage_id}"
            )
            invocation_key = f"{self.cur_microbatch}_{node.name}"
            start_ts = time.time()
            forward_name = event_name(
                Phase.FORWARD, stage_id, self.cur_microbatch
            )
            forward_id = event_id(
                Phase.FORWARD, stage_id, self.cur_microbatch, self.batch_id
            )
            name = f"I{forward_name}"
            id = f"I{forward_id}"
            stage_executor.rpc_async().invoke(
                invocation_key,
                Phase.FORWARD,
                args,
                kwargs,
                self.cur_microbatch,
                debug_str=node.format_node(),
                output_refcount=len(users),
                batch_id=self.batch_id,
                num_microbatches=self.num_microbatches,
            )
            finish_ts = time.time()
            self.record_event(
                rank=0,
                start_ts=start_ts,
                finish_ts=finish_ts,
                id=id,
                name=name,
                type="invoke",
                mbid=self.cur_microbatch,
            )
            self.record_event_dependency(
                from_id=name, to_id=f"R{forward_name}", type="invoke"
            )
            return ValueReference(stage_id, invocation_key)
        else:
            logging.debug(
                f"[root][{self.cur_microbatch}] Running local operation {target} from driver"
            )
            return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        node = self.node_list[self.pc]
        invocation_key = f"{self.cur_microbatch}_{node.name}"
        # if PipelineDriver is running inside `torch.no_grad()` context manager then `stage_backward*` nodes
        # are excluded from execution, so we need exclude `stage_backward*` from reference count, otherwise
        # it will cause memory leak.
        users = (
            list(
                filter(
                    lambda user: not user.name.startswith("stage_backward"),
                    node.users.keys(),
                )
            )
            if not torch.is_grad_enabled()
            else node.users.keys()
        )
        if target is operator.getitem and isinstance(args[0], ValueReference):
            val_ref = args[0]
            stage_id = val_ref.stage_id
            num_users = len(users)
            if not torch.is_grad_enabled() and val_ref.unique_key == "noop":
                return ValueReference(stage_id, "noop")
            elif num_users == 0:
                # TODO: investigate why there are getitem calls with 0 users
                return ValueReference(stage_id, "noop")
            else:
                indices = self.stage_output_indices.setdefault(stage_id, [])
                arg_idx = args[1]
                index_tuple = (invocation_key, num_users, val_ref, arg_idx)
                logging.debug(
                    f"[root][{self.cur_microbatch}] Appending getitem tuple to stage {stage_id}: {index_tuple}"
                )
                indices.append(index_tuple)
                return ValueReference(stage_id, invocation_key)
        elif target is stage_backward:
            assert "fw_stage" in node.meta
            stage_id, stage_executor = self.remote_stage_executor_rrefs[
                node.meta["fw_stage"]
            ]
            if torch.is_grad_enabled():
                logging.debug(
                    f"[root][{self.cur_microbatch}] Issuing BW invocation "
                    f'for target {node.meta["fw_stage"]} on stage {stage_id}'
                )
                start_ts = time.time()
                backward_name = event_name(
                    Phase.BACKWARD, stage_id, self.cur_microbatch
                )
                backward_id = event_id(
                    Phase.BACKWARD, stage_id, self.cur_microbatch, self.batch_id
                )
                name = f"I{backward_name}"
                id = f"I{backward_id}"
                stage_executor.rpc_async().invoke(
                    invocation_key,
                    Phase.BACKWARD,
                    args,
                    kwargs,
                    self.cur_microbatch,
                    debug_str=node.format_node(),
                    output_refcount=len(users),
                    batch_id=self.batch_id,
                    num_microbatches=self.num_microbatches,
                )
                finish_ts = time.time()
                self.record_event(
                    rank=0,
                    start_ts=start_ts,
                    finish_ts=finish_ts,
                    id=id,
                    name=name,
                    type="invoke",
                    mbid=self.cur_microbatch,
                )
                self.record_event_dependency(
                    from_id=name, to_id=backward_name, type="invoke"
                )
                return ValueReference(stage_id, invocation_key)
            else:
                return ValueReference(stage_id, "noop")
        elif target is sync_barrier:
            executor_keys = list(self.remote_stage_executor_rrefs.keys())
            stage_id, stage_executor = self.remote_stage_executor_rrefs[
                executor_keys[0]
            ]
            logging.debug(
                f"[root][{self.cur_microbatch}] Issuing sync invocation "
                f"on stage {stage_id}"
            )
            stage_executor.rpc_async().invoke(
                invocation_key,
                Phase.SYNC_BARRIER,
                args,
                kwargs,
                self.cur_microbatch,
                debug_str=node.format_node(),
                output_refcount=len(users),
                batch_id=self.batch_id,
                num_microbatches=self.num_microbatches,
            )
            return ValueReference(stage_id, invocation_key)
        elif target is _null_coalesce_accumulate:
            assert "fw_stage" in node.meta
            stage_id, stage_executor = self.remote_stage_executor_rrefs[
                node.meta["fw_stage"]
            ]
            if torch.is_grad_enabled():
                logging.debug(
                    f"[root][{self.cur_microbatch}] Issuing accumulate grad invocation "
                    f'for target {node.meta["fw_stage"]} on stage {stage_id}'
                )
                stage_executor.rpc_async().invoke(
                    invocation_key,
                    Phase.ACCUMULATE_GRAD,
                    args,
                    kwargs,
                    self.cur_microbatch,
                    debug_str=node.format_node(),
                    output_refcount=len(users),
                    batch_id=self.batch_id,
                    num_microbatches=self.num_microbatches,
                )
                return ValueReference(stage_id, invocation_key)
            else:
                return ValueReference(stage_id, "noop")
        else:
            raise AssertionError(f"Unknown operator {torch.typename(target)}")

    def issue_coalesced_getitem_calls(self):
        if len(self.stage_output_indices) == 0:
            return
        logging.debug(
            f"[root][{self.cur_microbatch}] Issuing getitem calls to stage: {self.stage_output_indices.keys()}"
        )

        for stage_id in self.stage_output_indices:
            stage_executor = self.stage_to_executor[stage_id]
            name = f"G{stage_id},{self.cur_microbatch}"
            id = f"{name},{self.batch_id}"
            start_ts = time.time()
            stage_executor.rpc_async().coalesced_index_value(
                self.stage_output_indices[stage_id]
            )
            finish_ts = time.time()
            self.record_event(
                rank=0,
                start_ts=start_ts,
                finish_ts=finish_ts,
                id=id,
                name=name,
                type="invoke",
                mbid=self.cur_microbatch,
            )

        self.stage_output_indices.clear()

    def run_until(
        self, predicate: Callable[[pippy.fx.Node], bool]
    ) -> Optional[pippy.fx.Node]:
        while self.pc < len(self.node_list):
            node = self.node_list[self.pc]

            if predicate(node):
                # Issue coalesced getitem calls as we pause issuing stage calls
                self.issue_coalesced_getitem_calls()
                return node

            self.run_one(node)

        # Have run through the entire node_list, using None to mean no node left to run
        return None

    def run_one(self, node):
        # TODO: hoist run() implementation
        logging.debug(
            f"[{self.cur_microbatch}] Issue command to run {node.format_node()}"
        )
        self.env[node] = super().run_node(node)

        # TODO: we could potentially move this waiting to the use sites for an RRef
        # (i.e. during Interpreter.map_nodes_to_values or when we pass args/kwargs
        #  to the callees) as an optimization
        # TODO: is it possible for there to be a blocking version of this API?
        def wait_for_confirmation(n):
            # The following if will not be true as we are using our own ValueRef
            # instead of RPC's RRef
            if isinstance(n, torch._C._distributed_rpc.PyRRef):
                while not n.confirmed_by_owner():
                    pass

        pippy.fx.node.map_aggregate(self.env[node], wait_for_confirmation)

        if DEBUG and isinstance(
            self.env[node], torch._C._distributed_rpc.PyRRef
        ):
            print(node, self.env[node])
            self.env[node].to_here()

        # Insert tensor meta to ValueReference returned by node call
        # TODO: there is some problem with "call_function", disabling for now
        if node.op == "call_module":
            if "tensor_meta" in node.meta and isinstance(
                node.meta["tensor_meta"],
                shape_prop.TensorMetadata,
            ):
                val_ref: ValueReference = self.env[node]
                val_ref.meta.setdefault("tensor_meta", node.meta["tensor_meta"])

        self.pc += 1
        return node

    def propagate_shape(self, args, kwargs):
        logging.info("Propagating shape across split GraphModule")
        sp = shape_prop.ShapeProp(self.module)
        # Not sure why FX's propagate API takes only args. Hence we unpack kwargs.values() without keys here
        sp.propagate(*args, *kwargs.values())
        for node in self.node_list:
            logging.debug(f"Node: {node.name}, outputs: ")
            if "tensor_meta" in node.meta:
                if isinstance(
                    node.meta["tensor_meta"], shape_prop.TensorMetadata
                ):
                    logging.debug(f"- {node.meta['tensor_meta']}")
                else:
                    # Multiple output tensors
                    for t_meta in node.meta["tensor_meta"]:
                        logging.debug(f"- {t_meta}")


class _run_until_criteria:
    def __init__(self):
        self.seen_stages = 0

    # Run the node we start with including all nodes that are tuple
    # indexing, then stop
    def hitting_next_stage(self, node):
        if node.op == "output":
            return True

        if (
            node.target != operator.getitem
            and node.target != _null_coalesce_accumulate
        ):
            self.seen_stages += 1

        if self.seen_stages > 1:
            return True
        elif self.seen_stages == 1 and node.target == _null_coalesce_accumulate:
            # We are hitting the accumulate call of the next (backward) stage, stop
            return True
        else:
            return False


class PipelineDriverFillDrain(PipelineDriverBase):
    def __init__(
        self,
        pipe: Pipe,
        chunks: int,
        world_size: int,
        all_ranks: List[int] = None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
        single_loss: bool = False,
        _debug_mask_minibatches: bool = False,
        max_outstanding=None,
        interleave_stages=False,
        _record_mem_dumps=False,
        checkpoint=False,
        use_c10d=False,
        loss_reducer: LossReducer = sum_reducer,
    ):
        super().__init__(
            pipe,
            chunks,
            world_size,
            all_ranks,
            args_chunk_spec,
            kwargs_chunk_spec,
            output_chunk_spec,
            _debug_mask_minibatches,
            max_outstanding=max_outstanding,
            interleave_stages=interleave_stages,
            _record_mem_dumps=_record_mem_dumps,
            checkpoint=checkpoint,
            use_c10d=use_c10d,
            loss_reducer=loss_reducer,
        )
        self.single_loss = single_loss

        self.last_grads = None

        self._init_remote_executors()

    def forward(self, *args, **kwargs):
        if self.single_loss:
            raise NotImplementedError("Single minibatch loss not implemented")

        # Roadmap:
        # 1) Micro-batch splitting - divide input arguments out into concrete chunk values
        # 2) Interpreter tiling - one interpreter per micro-batch
        # 3) Scheduling - Use control logic to advance interpreters to issue round-robin
        #       forward work items, then round-robin losses, then round-robin backwards

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            args,
            kwargs,
            self.chunks,
            self.args_chunk_spec,
            self.kwargs_chunk_spec,
            self._debug_mask_minibatches,
        )

        real_num_chunks = self.chunks
        if len(args_split) < self.chunks:
            real_num_chunks = len(args_split)
            warnings.warn(
                f"Reducing micro-batch numbers from {self.chunks} to "
                f"{real_num_chunks}."
            )

        logging.info(
            f"[root] Running pipeline with {real_num_chunks} micro-batches"
        )

        self.microbatch_interpreters = []

        batch_id = self.batch_id
        self.batch_id += 1

        for chunk in range(real_num_chunks):
            logging.debug(
                f"[root] Instantiating microbatch interpreter for chunk {chunk}"
            )
            interp = RemoteInterpreter(
                remote_stage_executor_rrefs=self.remote_stage_executor_rrefs,
                stage_to_executor=self.stage_to_executor,
                module=self.pipe.split_gm,
                cur_microbatch=chunk,
                args=args_split[chunk],
                kwargs=kwargs_split[chunk],
                batch_id=batch_id,
                num_microbatches=real_num_chunks,
            )
            # If user wants to use c10d for P2P, we would perform the shape propagation here. The shape prop is
            # performed per batch, thus supporting dynamic shape in batch dimension. Dynamic shape in microbatch
            # dimension is not yet supported, because all RemoteInterpreters share the same shape info (since they share
            # the same split_gm)
            if self.use_c10d and chunk == 0:
                interp.propagate_shape(args_split[chunk], kwargs_split[chunk])

            self.microbatch_interpreters.append(interp)

        logging.debug(
            f"[root] {len(self.microbatch_interpreters)} instantiated"
        )

        # Deterministic clock cycle - see torchgpipe paper section 3.2.1 for details

        # Advance past placeholders
        for interp in self.microbatch_interpreters:
            interp.run_until(lambda n: n.op != "placeholder")

        # Ramp-up, admit diagonal wavefront until we get to a full diagonal
        # location in the matrix

        for ramp_up_idx in range(len(self.microbatch_interpreters)):
            for i in range(ramp_up_idx + 1):
                interp = self.microbatch_interpreters[i]
                criteria = _run_until_criteria()
                interp.run_until(criteria.hitting_next_stage)

        # Steady-state. We have a full diagonal in the matrix; keep dispatching
        # across the diagonal

        any_valid = True
        while any_valid:
            any_valid = False
            for interp in self.microbatch_interpreters:
                start_node = interp.node_list[
                    min(interp.pc, len(interp.node_list) - 1)
                ]
                criteria = _run_until_criteria()
                interp.run_until(criteria.hitting_next_stage)

                any_valid |= interp.node_list[interp.pc] != start_node

        last_nodes = [
            interp.node_list[interp.pc]
            for interp in self.microbatch_interpreters
        ]
        assert all(node.op == "output" for node in last_nodes)

        local_results_and_last_grads = self._retrieve_output_values(
            self.microbatch_interpreters, last_nodes
        )

        if self.pipe.has_loss_and_backwards:
            # Shared parameter sync
            # At this point, all of the gradient jobs should have been run
            # (by way of the synchronization dependency earlier)
            self._sync_replicated_params()

        if DEBUG:
            self._check_stages_cleanup()

        if self.pipe.has_loss_and_backwards:
            local_results = []
            last_grads = []
            for local_result in local_results_and_last_grads:
                local_results.append(local_result[0])
                last_grads.append(local_result[1])

            self.last_grads = last_grads  # type: ignore[assignment]
        else:
            local_results = local_results_and_last_grads

        return merge_chunks(
            local_results, self.output_chunk_spec, self._debug_mask_minibatches
        )


class PipelineDriver1F1B(PipelineDriverFillDrain):
    def __init__(
        self,
        pipe: Pipe,
        chunks: int,
        world_size: int,
        all_ranks: List[int] = None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
        single_loss: bool = False,
        _debug_mask_minibatches: bool = False,
        interleave_stages=False,
        _record_mem_dumps=False,
        checkpoint=False,
        use_c10d=False,
        loss_reducer: LossReducer = sum_reducer,
    ):
        # In 1F1B with backward stages, the maximum number of outstanding
        # micro-batches equals the number of pipeline stages
        max_outstanding = (
            pipe.num_stages if pipe.has_loss_and_backwards else None
        )

        super().__init__(
            pipe,
            chunks,
            world_size,
            all_ranks,
            args_chunk_spec,
            kwargs_chunk_spec,
            output_chunk_spec,
            single_loss,
            _debug_mask_minibatches,
            max_outstanding=max_outstanding,
            interleave_stages=interleave_stages,
            _record_mem_dumps=_record_mem_dumps,
            checkpoint=checkpoint,
            use_c10d=use_c10d,
            loss_reducer=loss_reducer,
        )


class PipelineDriverInterleaved1F1B(PipelineDriver1F1B):
    def __init__(
        self,
        pipe: Pipe,
        chunks: int,
        world_size: int,
        all_ranks: List[int] = None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
        single_loss: bool = False,
        _debug_mask_minibatches: bool = False,
        _record_mem_dumps=False,
        checkpoint=False,
        use_c10d=False,
        loss_reducer: LossReducer = sum_reducer,
    ):
        super().__init__(
            pipe,
            chunks,
            world_size,
            all_ranks,
            args_chunk_spec,
            kwargs_chunk_spec,
            output_chunk_spec,
            single_loss,
            _debug_mask_minibatches,
            interleave_stages=True,
            _record_mem_dumps=_record_mem_dumps,
            checkpoint=checkpoint,
            use_c10d=use_c10d,
            loss_reducer=loss_reducer,
        )
