# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import logging
import os
import socket

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from pippy.IR import Pipe, TrivialLossWrapper, pipe_split, _null_coalesce_accumulate
from pippy.PipelineDriver import PipelineDriverBase, PipelineDriverFillDrain, PipelineDriver1F1B
from pippy.microbatch import TensorChunkSpec, CustomReducer

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
}

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

torch.fx.Tracer.proxy_buffer_attributes = True


def run_master(args):
    all_ranks = list(range(1, args.world_size))  # exclude master rank = 0
    chunks = len(all_ranks)
    bs = 4 * chunks
    hid_dim = 50

    class Code(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(hid_dim, hid_dim)

        def forward(self, x):
            x = self.linear(x)
            pipe_split()
            y = torch.relu(x)
            pipe_split()
            z = torch.sigmoid(x)
            pipe_split()
            return y + z

    c = Code()
    c.train()
    mse_loss = torch.nn.MSELoss()
    wrapper = TrivialLossWrapper(c, mse_loss)
    accum_pipe = Pipe.from_tracing(wrapper)
    assert 4 == len(list(accum_pipe.split_gm.children()))
    assert any(n.target == _null_coalesce_accumulate for n in accum_pipe.split_gm.graph.nodes)

    input = torch.randn(bs, hid_dim)
    target = torch.randn(bs, hid_dim)
    accum_pipe(input, target)

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}
    output_chunk_spec = CustomReducer(torch.tensor(0.0), lambda a, b: a + b)
    pipe_driver: PipelineDriverBase = schedules[args.schedule](accum_pipe, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec, args.world_size - 1,
                                                               all_ranks=all_ranks, _debug_mask_minibatches=True)

    pipe_driver.run(chunks, input, target)


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
    args = parser.parse_args()

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
