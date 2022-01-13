from IR import Pipe, MultiUseParameterConfig, pipe_split
import torch
import torch.fx
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import operator
import logging
import threading
import queue

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

PROFILING_ENABLED = True

import torch.distributed.rpc as rpc

# logging.getLogger().setLevel(logging.INFO)

def to_here(a):
    if isinstance(a, torch._C._distributed_rpc.PyRRef):
        return a.to_here()
    else:
        return a

# LOG EOD 12/28:
#
# Figure out if we can use `Future` directly to pass future values between the
# ppeline stages and handle scheduling manually. Don't think we can put futures
# over the wire (see future_test.py). Instead, should we push values via RPC?
#
# LOG EOD 1/4:
# This seems fairly hairy. Future is returned directly by RPC agents that override
# send(). Hypothetically we could create an agent where on the callee, requests are
# handled by a scheduler. OTOH, we can probably just create some sort of message
# format that includes (microbatch_id, phase, identifier for input) and send that
# around.

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

    class WorkItem(NamedTuple):
        args : Tuple[Any]
        kwargs : Dict[str, Any]
        future : torch.futures.Future

    def __init__(self, mod):
        logging.info(f'Rank {local_rank} Instantiating PipeStageExecutor for module {mod}')
        self.mod = mod

        # NB: queue.Queue is internally synchronized
        self.queue : queue.Queue[self.WorkItem] = queue.Queue()

        self.worker_thread = threading.Thread(target=self.worker_loop, name=f'worker_{self.mod}', daemon=True)
        self.worker_thread.start()

    def worker_loop(self):
        while True:
            args_rref, kwargs_rref, future = self.queue.get()

            args = torch.fx.node.map_aggregate(args_rref, to_here)
            kwargs = torch.fx.node.map_aggregate(kwargs_rref, to_here)
            logging.info(f'rank {local_rank} running target {self.mod}')
            out_val = self.mod(*args, **kwargs)

            future.set_result(out_val)

    @rpc.functions.async_execution
    def invoke(self, args, kwargs):
        future = torch.futures.Future()
        self.queue.put(self.WorkItem(args, kwargs, future))
        return future

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

if local_rank == 0:
    d_hid = 512
    bs = 503

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.lin = torch.nn.Linear(d_hid, d_hid)

        def forward(self, x):
            x = torch.mm(x, self.mm_param)
            skip_connection = x
            x = torch.relu(x)
            pipe_split()
            x = torch.mm(x, self.mm_param)
            x = self.lin(x)
            pipe_split()
            x = torch.relu(x)
            x = x + skip_connection
            x = torch.mm(x, self.mm_param2)
            x = self.lin(x)
            return x

    ec = ExampleCode()
    ec(torch.randn(bs, d_hid))

    ec_pipe = Pipe.from_tracing(ec, MultiUseParameterConfig.TRANSMIT)

    def loss_code(x):
        return torch.sum(x)

    optimizer = torch.optim.SGD(ec_pipe.parameters(), 0.01)

    remote_stage_executor_rrefs : Dict[str, torch.distributed.rpc.RRef] = {}

    for rank, (name, mod) in enumerate(ec_pipe.split_gm.named_children()):
        remote_stage_executor_rrefs[name] = (rank, rpc.remote(rank, PipeStageExecutor, (mod,)))

    # Interpret top-level graph and issue remote calls

    class RemoteInterpreter(torch.fx.Interpreter):
        def __init__(self, remote_stage_executor_rrefs, module, garbage_collect_values = True):
            super().__init__(module, garbage_collect_values)
            self.remote_stage_executor_rrefs = remote_stage_executor_rrefs

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
                            new_tensor = torch.zeros_like(input)
                            new_tensor[start:finish] = input[start:finish]
                            chunk_tensors.append(new_tensor)
                    else:
                        chunk_tensors = torch.split(arg, sizes)

                    split_args.append(self.MicroBatchSplitTensor(chunk_tensors))

                    logging.info(f'Split tensor argument {i} into {chunks} chunks of sizes '
                                 f'{[chunk.shape for chunk in chunk_tensors]}')
                else:
                    split_args.append(arg)

            microbatch_results = []
            for chunk_idx in range(chunks):
                microbatch_args = []
                for arg in split_args:
                    microbatch_args.append(arg.chunks[chunk_idx] if isinstance(arg, self.MicroBatchSplitTensor) else arg)
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
                return stage_executor.remote().invoke(args, kwargs)
            else:
                logging.info(f'Running local operation {target} from driver')
                return super().call_module(target, args, kwargs)

        def call_function(self, target, args, kwargs):
            if target is operator.getitem and isinstance(args[0], torch._C._distributed_rpc.PyRRef):
                return args[0].remote().__getitem__(args[1])
            return super().call_function(target, args, kwargs)

    interp = RemoteInterpreter(remote_stage_executor_rrefs, ec_pipe.split_gm)

    input = torch.randn(bs, d_hid)

    check_numeric_equivalence = True

    # Warm up and correctness runs
    out = interp.run(input, chunks=5, _debug_mask_minibatches = True)
    ref_out = ec_pipe.split_gm(input)

    if check_numeric_equivalence:
        torch.testing.assert_allclose(out, ref_out)
        print(f'equivalence test passed {torch.sum(out)}')
        
    # Profiling runts
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        out = interp.run(input, chunks=5, _debug_mask_minibatches = False)
        ref_out = ec_pipe.split_gm(input)
        print(f'profiling run completed {torch.sum(ref_out)}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace('pipe.csv')

rpc.shutdown()
