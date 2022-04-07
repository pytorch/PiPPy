# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import socket
import time
from collections import defaultdict
from typing import List, Dict, Any
import logging

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.autograd import Function

from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split, TrivialLossWrapper
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, Event, Phase, PipelineDriverBase
from pippy.microbatch import TensorChunkSpec, CustomReducer
from pippy.visualizer import events_to_json

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
}

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

# WAR for SEV remediation https://github.com/pytorch/pytorch/commit/2337d4e5036a87f473decd2b1f6fe0439499902c
torch.fx.Tracer.proxy_buffer_attributes = True


@torch.fx.wrap
def sleep(x, t=1.0):
    time.sleep(t)
    return x


class SlowMSELoss(nn.MSELoss):
    def forward(self, input, target):
        return super().forward(sleep(input, t=0.1), target)


# Inherit from Function
class MyLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        # print("my forward")
        input = sleep(input)
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # print("my backward")
        grad_output = sleep(grad_output)
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


@torch.fx.wrap
def linear(input, weight, bias):
    return MyLinearFunction.apply(input, weight, bias)


class MyLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(MyLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return linear(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


def run_main(args):
    d_hid = 100
    bs = 400
    chunks = 4

    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = MyLinear(d_hid, d_hid)
            self.l2 = MyLinear(d_hid, d_hid)
            self.l3 = MyLinear(d_hid, d_hid)
            self.l4 = MyLinear(d_hid, d_hid)

        def forward(self, x):
            x = self.l1(x)
            pipe_split()
            x = self.l2(x)
            pipe_split()
            x = self.l3(x)
            pipe_split()
            x = self.l4(x)
            return x

    ec = ExampleCode()

    mse_loss = SlowMSELoss(reduction='sum')
    wrapper = TrivialLossWrapper(ec, mse_loss)
    ec_pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}
    output_chunk_spec = CustomReducer(torch.tensor(0.0), lambda a, b: a + b)

    pipe_driver: PipelineDriverBase = PipelineDriverFillDrain(ec_pipe, args_chunk_spec, kwargs_chunk_spec,
                                                              output_chunk_spec,
                                                              args.world_size, _debug_mask_minibatches=True)

    input = torch.randn(bs, d_hid)
    target = torch.randn(bs, d_hid)

    pipe_visualized_filename = "pipe_visualized.json"
    all_events = []
    for i in range(1):
        pipe_driver.run(chunks, input, target)
        events = pipe_driver.retrieve_events()
        # check_events_for_single_batch(events, args.world_size, chunks, pipe_visualized_filename)
        all_events.extend(events)
    with open(pipe_visualized_filename, "w") as f:
        f.write(events_to_json(all_events))


def check_events_for_single_batch(events: List[Event], all_ranks: int, chunks: int, pipe_visualized_filename: str):
    events_by_type_by_rank_by_mbid: Dict[Any, Dict[Any, Dict[Any, Event]]] = \
        defaultdict(lambda: defaultdict(lambda: dict()))
    for event in events:
        events_by_type_by_rank_by_mbid[event.type][event.rank][event.mbid] = event

    # Basic happens-before cross rank checks
    for rank in range(all_ranks - 1):
        next_rank = rank + 1
        for mbid in range(chunks):
            rank_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][rank][mbid]
            next_rank_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][next_rank][mbid]
            # happens-before cross-rank forward check
            assert next_rank_forward.start_ts >= rank_forward.finish_ts, \
                f"{rank_forward.name}({rank_forward.finish_ts}) must happen before " \
                f"{next_rank_forward.name}({next_rank_forward.start_ts}), see {pipe_visualized_filename}"

            rank_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][next_rank][mbid]
            next_rank_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][rank][mbid]
            # happens-before cross-rank backward check
            assert next_rank_backward.start_ts >= rank_backward.finish_ts, \
                f"{rank_backward.name}({rank_backward.finish_ts}) must happen before " \
                f"{next_rank_backward.name}({next_rank_backward.start_ts}), see {pipe_visualized_filename}"

    # Basic happens-before cross-microbatch checks
    for mbid in range(chunks - 1):
        next_mbid = mbid + 1
        for rank in range(all_ranks):
            rank_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][rank][mbid]
            next_mbid_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][rank][next_mbid]
            # happens-before cross-microbatch forward check
            assert next_mbid_forward.start_ts >= rank_forward.finish_ts, \
                f"{rank_forward.name}({rank_forward.finish_ts}) must happen before " \
                f"{next_mbid_forward.name}({next_mbid_forward.start_ts}), see {pipe_visualized_filename}"

            rank_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][rank][mbid]
            next_mbid_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][rank][next_mbid]
            # happens-before cross-microbatch backward check
            assert next_mbid_backward.start_ts >= rank_backward.finish_ts, \
                f"{rank_backward.name}({rank_backward.finish_ts}) must happen before " \
                f"{next_mbid_backward.name}({next_mbid_backward.start_ts}), see {pipe_visualized_filename}"

    # Overlap checks
    for mbid in range(chunks - 1):
        next_mbid = mbid + 1
        last_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][all_ranks - 1][mbid]
        first_next_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][0][next_mbid]
        # cross-microbatch forward overlap check
        # TODO: INVERT CONDITION TO >= AFTER OVERLAP FIX!!!
        assert last_forward.finish_ts <= first_next_forward.start_ts, \
            f"Forward microbatch {mbid} doesn't overlap with next microbatch {next_mbid}"

        last_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][all_ranks - 1][mbid]
        first_next_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][0][mbid + 1]
        # cross-microbatch forward overlap check
        # TODO: INVERT CONDITION TO >= AFTER OVERLAP FIX!!!
        assert last_backward.finish_ts <= first_next_backward.start_ts, \
            f"Backward microbatch {mbid} doesn't overlap with next microbatch {next_mbid}"


def run_worker(rank, world_size, args):
    print(f"rank = {rank} host/pid = {socket.gethostname()}/{os.getpid()}")
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    if rank == 0:
        run_main(args)
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    args = parser.parse_args()

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
