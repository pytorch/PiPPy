import torch
import torch.fx
import torch.distributed.rpc as rpc
from pippy.IR import Pipe, stage_backward
from pippy.microbatch import split_args_kwargs_into_chunks, merge_chunks
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Optional
import logging
import operator
import threading
import warnings
from inspect import Parameter, Signature

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
    LOSS = 1
    BACKWARD = 2

# TODO: do we need this?
class SchedState(Enum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    DONE = 3

class WorkItem:
    def __init__(
            self, phase, args, kwargs, future, microbatch_id, blocked_args_count, ready_args,
            state=SchedState.WAITING, debug_str=''):
        args_to_fwd = ['phase', 'args', 'kwargs', 'future', 'microbatch_id', 'blocked_args_count',
                       'ready_args', 'state', 'debug_str']

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

    def __str__(self):
        return f'WorkItem({self.debug_str})'

def _get_value_on_remote(caller_rank, callee_rank, runlist_key, microbatch, rref):
    logging.info(f'[{callee_rank}][{microbatch}] Executing async transfer of value '
                 f'{rref} initiated by rank {caller_rank} for {runlist_key}')
    return rref.local_value()

@rpc.functions.async_execution
def async_transfer(rank, microbatch, scheduler_rref, rref_arg, arg_idx, runlist_key):
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
                    work_item.state = SchedState.READY
                    self.ready_runlist[runlist_key] = self.waiting_runlist.pop(runlist_key)
                    self.ready_runlist_cv.notify()
                logging.info(f'[{rank}][{microbatch}] All operands ready')
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

    def __init__(self, mod, local_rank, loss_mod=None):
        self.local_rank = local_rank
        logging.info(f'[{self.local_rank}] Instantiating PipeStageExecutor')
        self.mod = mod
        self.loss_mod = loss_mod

        self.waiting_runlist_lock = threading.Lock()
        # self.waiting_runlist (*and the contained WorkItems*) are guarded by
        # self.waiting_runlist_lock
        self.waiting_runlist : Dict[WorkItem, None] = {}

        self.ready_runlist_lock = threading.Lock()
        self.ready_runlist_cv = threading.Condition(self.ready_runlist_lock)
        self.ready_runlist : Dict[str, WorkItem] = {}

        self.worker_thread = threading.Thread(target=self.worker_loop, name=f'worker_{self.mod}', daemon=True)
        self.worker_thread.start()

        # map microbatch ID to list of forward tensor args
        self.fwd_cache : Dict[int, Tuple[Any, List[torch.Tensor]]] = {}
        self.loss_cache : Dict[int, Tuple[Any, List[torch.Tensor]]] = {}

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
                cache = self.loss_cache if len(self.loss_cache) > 0 else self.fwd_cache
                kwargs = dict(kwargs)
                kwargs['stage_output'], kwargs['input_values'] = cache.pop(microbatch_id)

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
    def __init__(self, pipe : Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size : int,
                 all_ranks : List[int] = None):
        self.pipe = pipe
        self.world_size = world_size
        self.all_ranks = all_ranks
        self.executor_class = PipeStageExecutor
        self.args_chunk_spec = args_chunk_spec
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.output_chunk_spec = output_chunk_spec

        self._init_remote_executors()

    def _init_remote_executors(self):
        self.remote_stage_executor_rrefs : Dict[str, (int, torch.distributed.rpc.RRef)] = {}

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
                               f'{self.world_size}. Please ensure world_size is large enough to accommodate your pipeline.')

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

class RemoteInterpreter(torch.fx.Interpreter):
    def __init__(self, remote_stage_executor_rrefs, module, cur_microbatch : int,   
                 args, kwargs, garbage_collect_values=True):
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

    def call_module(self, target, args, kwargs):
        assert isinstance(target, str)
        node = self.node_list[self.pc]

        if target in self.remote_stage_executor_rrefs:
            rank, stage_executor = self.remote_stage_executor_rrefs[target]
            phase = Phase.LOSS if target == '_loss' else Phase.FORWARD
            logging.info(f'[root][{self.cur_microbatch}] Issuing {phase} '
                         f'invocation for target {target} on rank {rank}')
            invocation_key = f'{self.cur_microbatch}_{node.name}'
            return stage_executor.remote().invoke(
                invocation_key, phase, args, kwargs, self.cur_microbatch, debug_str=node.format_node())
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
                invocation_key, Phase.BACKWARD, args, kwargs, self.cur_microbatch, debug_str=node.format_node())
        return super().call_function(target, args, kwargs)

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
    def __init__(self, pipe : Pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size : int,
                 all_ranks : List[int] = None, single_loss : bool = False):
        super().__init__(pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size, all_ranks)
        self.single_loss = single_loss

    def run(self, args, kwargs, chunks : int, _debug_mask_minibatches : bool = False):
        if self.single_loss:
            raise NotImplementedError('Single minibatch loss not implemented')

        logging.info('[root] Running pipeline')
        # Roadmap:
        # 1) Micro-batch splitting - divide input arguments out into concrete chunk values
        # 2) Interpreter tiling - one interpreter per micro-batch
        # 3) Scheduling - Use control logic to advance interpreters to issue round-robin
        #       forward work items, then round-robin losses, then round-robin backwards

        args_split, kwargs_split = split_args_kwargs_into_chunks(args, kwargs, self.args_chunk_spec,
                                                                 self.kwargs_chunk_spec, chunks,
                                                                 _debug_mask_minibatches)

        microbatch_interpreters : List[self.RunUntilInterpreter] = []

        for chunk in range(chunks):
            logging.info(f'[root] Instantiating microbatch interpreter for chunk {chunk}') 
            interp = RemoteInterpreter(self.remote_stage_executor_rrefs, self.pipe.split_gm, chunk, args_split[chunk],
                                       kwargs_split[chunk])
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
            logging.info('[root] Program does not have loss/backward, returning outputs directly')
            # Forward-only; return output values
            return self._retrieve_output_values(microbatch_interpreters, last_nodes, _debug_mask_minibatches)

        logging.info('[root] Executing loss + backward stages')
        last_nodes = []
        for interp in microbatch_interpreters:
            last_nodes.append(interp.run_until(lambda n: n.op == 'output'))

        assert all(n == last_nodes[0] for n in last_nodes)
        assert last_nodes[0].op == 'output'
        return self._retrieve_output_values(microbatch_interpreters, last_nodes, _debug_mask_minibatches)

    def _retrieve_output_values(self, microbatch_interpreters, last_nodes, _debug_mask_minibatches):
        logging.info(f'[root] Combining output values from {len(microbatch_interpreters)} chunks')
        output_vals = []
        for interp, last_node in zip(microbatch_interpreters, last_nodes):
            interp.run_until(lambda n : False)
            output_vals.append(interp.env[last_node])

        local_results = torch.fx.node.map_aggregate(output_vals, to_here)

        return merge_chunks(local_results, self.output_chunk_spec, _debug_mask_minibatches)
