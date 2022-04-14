# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import socket
from typing import Dict

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverBase
from pippy.microbatch import TensorChunkSpec

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
}

torch.fx.Tracer.proxy_buffer_attributes = True


def run_master(args):
    d_hid = 512
    bs = 503

    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')

    print("Using schedule:", args.schedule)

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.lin = torch.nn.Linear(d_hid, d_hid)
            self.register_buffer('buffer', torch.randn(bs + 100, d_hid))

        def forward(self, x):
            x = torch.mm(x, self.mm_param)
            skip_connection = x
            x = torch.relu(x)
            pipe_split()
            x = torch.mm(x, self.mm_param) + self.buffer[:x.shape[0]]
            x = self.lin(x)
            pipe_split()
            x = torch.relu(x)
            x = x + skip_connection
            x = torch.mm(x, self.mm_param2)
            x = self.lin(x)
            return {'out': x}

    ec = ExampleCode()
    ec.to(args.device)
    ec_input = torch.randn(bs, d_hid, device=args.device)
    ec(ec_input)

    ec_pipe = Pipe.from_tracing(ec, MULTI_USE_PARAM_CONFIG)
    print(ec_pipe.split_gm)

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec: Dict = {}
    output_chunk_spec = {'out': TensorChunkSpec(0)}

    pipe_driver: PipelineDriverBase = schedules[args.schedule](ec_pipe, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec,
                                                               args.world_size, _debug_mask_minibatches=True)

    # # Warm up and correctness runs
    out = pipe_driver.run(5, ec_input)
    ref_out = ec_pipe(ec_input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_close(out['out'], ref_out['out'])
        print(f'equivalence test passed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}')

    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        out = pipe_driver.run(5, ec_input)
        ref_out = ec_pipe(ec_input)
        print(f'profiling run completed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace(f'{os.path.splitext(os.path.basename(__file__))[0]}.json')


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256,
                                              _transports=["shm", "uv"])
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
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 3)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    args = parser.parse_args()

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
