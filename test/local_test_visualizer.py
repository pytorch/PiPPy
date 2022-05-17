# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import logging
import os
import socket
import time
from collections import defaultdict
from functools import reduce
from typing import List, Dict, Any

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.autograd import Function

from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split, TrivialLossWrapper
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, Phase, PipelineDriverBase, \
    EventsContext, PipelineDriverInterleaved1F1B
from pippy.events import Event
from pippy.microbatch import TensorChunkSpec, CustomReducer
from pippy.visualizer import events_to_json
from test_commons import tp_transports # type: ignore

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

torch.fx.Tracer.proxy_buffer_attributes = True


@torch.fx.wrap
def sleep(x, t=1.0):
    time.sleep(t)
    return x


class SlowMSELoss(nn.MSELoss):
    def forward(self, input, target):
        return super().forward(sleep(input, t=0.01), target)


# Inherit from Function
class MyLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        # print("my forward")
        input = sleep(input, t=0.1)
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # print("my backward")
        grad_output = sleep(grad_output, t=0.3)
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


def run_master(args):
    d_hid = 100
    bs = 400
    chunks = 4
    batches = 1

    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)

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
    ec.to(args.device)

    mse_loss = SlowMSELoss(reduction='sum')
    wrapper = TrivialLossWrapper(ec, mse_loss)
    ec_pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}
    output_chunk_spec = CustomReducer(torch.tensor(0.0), lambda a, b: a + b)

    all_ranks = list(range(1, args.world_size))  # exclude master rank = 0
    pipe_driver: PipelineDriverBase = schedules[args.schedule](ec_pipe, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec, args.world_size - 1,
                                                               all_ranks=all_ranks,
                                                               _debug_mask_minibatches=True,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint))

    ec_input = torch.randn(bs, d_hid, device=args.device)
    target = torch.randn(bs, d_hid, device=args.device)

    pipe_visualized_filename = "pipe_visualized.json"
    batches_events_contexts = []
    for i in range(batches):
        pipe_driver.chunks = chunks
        pipe_driver(ec_input, target)
        batches_events_contexts.append(pipe_driver.retrieve_events())

    # first: save file
    all_events_contexts: EventsContext = reduce(lambda c1, c2: EventsContext().update(c1).update(c2),
                                                batches_events_contexts, EventsContext())
    with open(pipe_visualized_filename, "w") as f:
        f.write(events_to_json(all_events_contexts))

    # TODO: Investigate flakiness! TODO(https://github.com/pytorch/PiPPy/issues/136)
    # # second: perform checks
    # for events_context in batches_events_contexts:
    #     check_events_for_single_batch(events_context.events, all_ranks, chunks, pipe_visualized_filename)


def check_events_for_single_batch(events: List[Event], all_stages: List[int], chunks: int,
                                  pipe_visualized_filename: str):
    events_by_type_by_rank_by_mbid: Dict[Any, Dict[Any, Dict[Any, Event]]] = \
        defaultdict(lambda: defaultdict(lambda: dict()))
    for event in events:
        events_by_type_by_rank_by_mbid[event.type][event.rank][event.mbid] = event

    def start_ts(e: Event, eps=0.1):
        return e.start_ts + (e.finish_ts - e.start_ts) * eps

    def finish_ts(e: Event, eps=0.1):
        return e.finish_ts - (e.finish_ts - e.start_ts) * eps

    # Basic happens-before cross rank checks
    for i in range(len(all_stages) - 1):
        rank = all_stages[i]
        next_rank = all_stages[i + 1]
        for mbid in range(chunks):
            rank_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][rank][mbid]
            next_rank_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][next_rank][mbid]
            # happens-before cross-rank forward check
            assert start_ts(next_rank_forward) >= finish_ts(rank_forward), \
                f"{rank_forward.name}({rank_forward.finish_ts}) must happen before " \
                f"{next_rank_forward.name}({next_rank_forward.start_ts}), see {pipe_visualized_filename}"

            rank_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][next_rank][mbid]
            next_rank_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][rank][mbid]
            # happens-before cross-rank backward check
            assert start_ts(next_rank_backward) >= finish_ts(rank_backward), \
                f"{rank_backward.name}({rank_backward.finish_ts}) must happen before " \
                f"{next_rank_backward.name}({next_rank_backward.start_ts}), see {pipe_visualized_filename}"

    # Basic happens-before cross-microbatch checks
    for mbid in range(chunks - 1):
        next_mbid = mbid + 1
        for i in range(len(all_stages) - 1):
            rank = all_stages[i]
            rank_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][rank][mbid]
            next_mbid_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][rank][next_mbid]
            # happens-before cross-microbatch forward check
            assert start_ts(next_mbid_forward) >= finish_ts(rank_forward), \
                f"{rank_forward.name}({rank_forward.finish_ts}) must happen before " \
                f"{next_mbid_forward.name}({next_mbid_forward.start_ts}), see {pipe_visualized_filename}"

            rank_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][rank][mbid]
            next_mbid_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][rank][next_mbid]
            # happens-before cross-microbatch backward check
            assert start_ts(next_mbid_backward) >= finish_ts(rank_backward), \
                f"{rank_backward.name}({rank_backward.finish_ts}) must happen before " \
                f"{next_mbid_backward.name}({next_mbid_backward.start_ts}), see {pipe_visualized_filename}"

    # Overlap checks
    for mbid in range(chunks - 1):
        next_mbid = mbid + 1
        last_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][all_stages[-1]][mbid]
        first_next_forward = events_by_type_by_rank_by_mbid[Phase.FORWARD][all_stages[0]][next_mbid]
        # cross-microbatch forward overlap check
        assert last_forward.finish_ts >= first_next_forward.start_ts, \
            f"Forward microbatch {mbid} doesn't overlap with next microbatch {next_mbid}"

        last_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][all_stages[0]][mbid]
        first_next_backward = events_by_type_by_rank_by_mbid[Phase.BACKWARD][all_stages[-1]][next_mbid]
        # cross-microbatch forward overlap check
        assert last_backward.finish_ts >= first_next_backward.start_ts, \
            f"Backward microbatch {mbid} doesn't overlap with next microbatch {next_mbid}"


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256,
                                              rpc_timeout=1800,
                                              _transports=tp_transports())
    if args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
            for i in range(world_size):
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
        else:
            args.cuda = 0
    args.device = f'cuda:{dev_id}' if args.cuda else 'cpu'
    print(f"rank = {rank} host/pid/device = "
          f"{socket.gethostname()}/{os.getpid()}/{args.device}")

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    if rank == 0:
        run_master(args)
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 5)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
