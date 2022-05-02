# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import copy
import logging
import os
import socket
from typing import Dict

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from pippy.IR import MultiUseParameterConfig, Pipe, TrivialLossWrapper, pipe_split
from pippy.PipelineDriver import (
    PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverBase, PipelineDriverInterleaved1F1B
)
from pippy.microbatch import TensorChunkSpec, CustomReducer
from test.test_commons import tp_transports

# TODOs for implementing forward/backward/loss with schedules:
# * ability to switch between full-batch loss vs. per-microbatch loss. shen mentioned
# this might change numerics. So we should have the ability to compute loss over
# the whole minibatch rather than doing it for each micro-batch

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B
}

VERBOSE = bool(os.environ.get('VERBOSE', False))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

def get_grad_from_executor(executor, qualname):
    mod = executor.local_value().mod
    if isinstance(mod, torch.nn.parallel.DistributedDataParallel):
        return mod.module.get_parameter(qualname).grad
    else:
        return mod.get_parameter(qualname).grad


torch.fx.Tracer.proxy_buffer_attributes = True

dp_pg_for_reference = None

def run_master(args, pp_ranks):
    torch.manual_seed(42)

    d_hid = 50
    bs = 503
    CHUNKS = 5
    DEBUG_MASK_MINIBATCHES = True
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')

    print("Using schedule:", args.schedule)

    def rand_zeros_or_ones(shape):
        return torch.randint(0, 2, shape).float()

    class ZeroOneLinear(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.w = torch.nn.Parameter(rand_zeros_or_ones((in_dim, out_dim)))

        def forward(self, x):
            return x @ self.w

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_param = torch.nn.Parameter(rand_zeros_or_ones((d_hid, d_hid)))
            self.mm_param2 = torch.nn.Parameter(rand_zeros_or_ones((d_hid, d_hid)))
            self.lin = ZeroOneLinear(d_hid, d_hid)
            self.register_buffer('buffer', 0.00001 * rand_zeros_or_ones((bs + 100, d_hid)))

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
            return x

    ec = ExampleCode()
    ec.to(args.device)
    ec(torch.randn(bs, d_hid, device=args.device))
    ec.train()

    # TODO: works with sum, need to define semantics for e.g. mean
    mse_loss = torch.nn.MSELoss(reduction='sum')
    wrapper = TrivialLossWrapper(ec, mse_loss)
    ec_pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)
    if args.rank == 0:
        print(ec_pipe.split_gm)

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec: Dict = {}
    output_chunk_spec = CustomReducer(torch.tensor(0.0), lambda a, b: a + b)

    pipe_driver : PipelineDriverBase = schedules[args.schedule](ec_pipe, args_chunk_spec, kwargs_chunk_spec,
                                                                output_chunk_spec, args.pp_group_size,
                                                                all_ranks=pp_ranks,
                                                                _debug_mask_minibatches=DEBUG_MASK_MINIBATCHES,
                                                                _record_mem_dumps=bool(args.record_mem_dumps),
                                                                checkpoint=bool(args.checkpoint))
    print(f'Rank {args.rank} Instantiated pipe with ranks {pp_ranks}')

    pipe_driver.init_data_parallel(dp_group_size=args.dp_group_size)

    torch.manual_seed(args.rank)
    input = torch.randn(bs, d_hid, device=args.device)
    target = torch.randn(bs, d_hid, device=args.device)

    # TODO: distributed optimizer
    out = pipe_driver.run(CHUNKS, input, target)

    print(f'Rank {args.rank} got loss value {out}')

    all_grad_qualnames = {k: None for k, v in ec_pipe.named_parameters()}

    pipe_grads = {}

    for name in all_grad_qualnames:
        assert 'split_gm.' in name
        _, module_name, param_qualname = name.split('.', maxsplit=2)

        assert module_name in pipe_driver.remote_stage_executor_rrefs
        rank, module_rref = pipe_driver.remote_stage_executor_rrefs[module_name]
        grad_value = rpc.rpc_sync(module_rref.owner(), get_grad_from_executor, (module_rref, param_qualname))
        pipe_grads[name] = copy.deepcopy(grad_value)

    wrapper_ddp = torch.nn.parallel.DistributedDataParallel(wrapper, process_group=dp_pg_for_reference)

    optim = torch.optim.SGD(wrapper_ddp.parameters(), lr=0.05)
    optim.zero_grad()
    with torch.autograd.profiler.profile(enabled=args.rank == 0) as prof:
        wrapper_out = wrapper_ddp(input, target)
        wrapper_out.backward()
    if prof:
        prof.export_chrome_trace('ref.json')

    not_close_grads = []
    ref_grads = {}

    for name in all_grad_qualnames:
        remapped_qualname = ec_pipe.remap_qualname(name)
        param = wrapper_ddp.module.get_parameter(remapped_qualname)
        assert name in pipe_grads, f'{name} not in pipe_grads keys {pipe_grads.keys()}'
        ref_grads[name] = copy.deepcopy(param.grad)
        if not torch.allclose(pipe_grads[name], ref_grads[name]):
            not_close_grads.append(name)

    if len(not_close_grads):
        raise AssertionError(f'Gradients not close: {not_close_grads}')

    print('Gradient equivalence test passed')


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

    # Init DDP process group
    backend = "nccl" if args.cuda else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )

    pp_ranks_per_dp_group = [[i * args.dp_group_size + rank for i in range(args.pp_group_size)]
                             for rank in range(args.dp_group_size)]

    global dp_pg_for_reference
    dp_pg_for_reference = torch.distributed.new_group(list(range(args.dp_group_size)))

    if rank >= 0 and rank // args.dp_group_size == 0:
        args.rank = rank
        run_master(args, pp_ranks_per_dp_group[rank])
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 12)))
    # in row-major
    # DP ranks are contiguous rows of size `args.dp_group_size`
    # PP ranks are non-contiguous columns of size `args.pp_group_size`
    #
    # if dp_group_size = 4 and pp_group_size = 3
    #
    #   0 1 2  3
    #   4 5 6  7
    #   8 9 10 11
    #
    # DP ranks are [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    # PP ranks are [0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]
    parser.add_argument('--dp_group_size', type=int, default=int(os.getenv("DP_GROUP_SIZE", 4)))
    parser.add_argument('--pp_group_size', type=int, default=int(os.getenv("PP_GROUP_SIZE", 3)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    assert args.dp_group_size * args.pp_group_size == args.world_size

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
