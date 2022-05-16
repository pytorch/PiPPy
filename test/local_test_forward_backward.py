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
from pippy.PipelineDriver import PipelineDriverBase, PipelineDriverFillDrain, PipelineDriver1F1B, \
    PipelineDriverInterleaved1F1B
from pippy.microbatch import TensorChunkSpec, CustomReducer, split_args_kwargs_into_chunks
from test_commons import tp_transports # type: ignore

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

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)


# import ctypes
# libc = ctypes.cdll.LoadLibrary("libc.so.6")
# libc.prctl.argtypes = [
#     ctypes.c_int,
#     ctypes.c_ulong,
#     ctypes.c_ulong,
#     ctypes.c_ulong,
#     ctypes.c_ulong,
# ]
# libc.prctl.restype = ctypes.c_int
# libc.prctl(0x59616D61, -1, 0, 0, 0)

def get_grad_from_executor(executor, qualname):
    return executor.local_value().mod.get_parameter(qualname).grad


def set_grad_in_executor(executor, qualname, value):
    param = executor.local_value().mod.get_parameter(qualname)
    param.grad = value


torch.fx.Tracer.proxy_buffer_attributes = True


def run_master(args):
    torch.manual_seed(42)

    d_hid = 50
    bs = 503
    CHUNKS = 5
    DEBUG_MASK_MINIBATCHES = True
    REF_USE_MICROBATCHES = True
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
            pipe_split()
            x = torch.relu(x)
            return x

    ec = ExampleCode()
    ec.to(args.device)
    ec_input = torch.randn(bs, d_hid, device=args.device)
    ec(ec_input)
    ec.train()

    # TODO: works with sum, need to define semantics for e.g. mean
    mse_loss = torch.nn.MSELoss(reduction='sum')
    wrapper = TrivialLossWrapper(ec, mse_loss)
    ec_pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)
    print(ec_pipe.split_gm)

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec: Dict = {}
    output_chunk_spec = CustomReducer(torch.tensor(0.0), lambda a, b: a + b)

    pipe_driver: PipelineDriverBase = schedules[args.schedule](ec_pipe, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec,
                                                               args.world_size,
                                                               _debug_mask_minibatches=DEBUG_MASK_MINIBATCHES,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint))

    target = torch.randn(bs, d_hid, device=args.device)

    # TODO: distributed optimizer
    pipe_driver.chunks = CHUNKS
    out = pipe_driver(ec_input, target)

    all_grad_qualnames = {k: None for k, v in ec_pipe.named_parameters()}

    pipe_grads = {}

    for name in all_grad_qualnames:
        assert 'split_gm.' in name
        _, module_name, param_qualname = name.split('.', maxsplit=2)

        assert module_name in pipe_driver.remote_stage_executor_rrefs
        stage_id, module_rref = pipe_driver.remote_stage_executor_rrefs[module_name]
        grad_value = rpc.rpc_sync(module_rref.owner(), get_grad_from_executor, (module_rref, param_qualname))
        pipe_grads[name] = copy.deepcopy(grad_value)

    optim = torch.optim.SGD(ec_pipe.split_gm.parameters(), lr=0.05)
    optim.zero_grad()
    if REF_USE_MICROBATCHES:
        args_split, kwargs_split = split_args_kwargs_into_chunks((ec_input, target), {}, args_chunk_spec,
                                                                 kwargs_chunk_spec, CHUNKS,
                                                                 DEBUG_MASK_MINIBATCHES)
        ref_outs = []
        for chunk in range(CHUNKS):
            ref_outs.append(ec_pipe(*args_split[chunk]))
        ref_out = torch.sum(torch.stack(ref_outs))
    else:
        ref_out = ec_pipe(ec_input, target)

    # Shared parameter sync for reference. TODO: move this to actual runtime
    for param_set in ec_pipe.replicated_params:
        grad_values = []
        for module_name, param_qualname in param_set.items():
            grad_values.append(ec_pipe.get_parameter(f'split_gm.{module_name}.{param_qualname}').grad)

        synced_value = torch.sum(torch.stack(grad_values), dim=0)

        for module_name, param_qualname in param_set.items():
            ec_pipe.get_parameter(f'split_gm.{module_name}.{param_qualname}').grad = synced_value

    # TODO: scale output
    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_close(out, ref_out)
        print(f'equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}')

    not_close_grads = []
    ref_grads = {}
    for name in all_grad_qualnames:
        param = ec_pipe.get_parameter(name)
        assert name in pipe_grads, f'{name} not in pipe_grads keys {pipe_grads.keys()}'
        ref_grads[name] = param.grad
        if not torch.allclose(pipe_grads[name], param.grad):
            not_close_grads.append(name)

    for name in not_close_grads:
        pipe_grad = pipe_grads[name]
        ref_grad = ref_grads[name]

        relative_delta = torch.abs(pipe_grad - ref_grad) / ref_grad
        assert False, f'Gradient for parameter {name} is not numerically close! Relative diff mean ' \
                      f'{torch.mean(relative_delta)} std {torch.std(relative_delta)} max {torch.max(relative_delta)}'

    print('Gradient equivalence test passed')

    # Test equivalence with initial code as well
    orig_optim = torch.optim.SGD(ec.parameters(), lr=0.05)
    orig_optim.zero_grad()
    orig_loss = mse_loss(ec(ec_input), target)
    orig_loss.backward()
    torch.testing.assert_close(out, orig_loss)

    not_close_orig_grads = []
    not_found_mappings = []

    for name in all_grad_qualnames:
        try:
            remapped_qualname = ec_pipe.remap_qualname(name)
        except KeyError:
            not_found_mappings.append(name)
        else:
            orig_grad = wrapper.get_parameter(remapped_qualname).grad
            pipe_grad = pipe_grads[name]
            if not torch.allclose(pipe_grad, orig_grad):
                not_close_orig_grads.append(name)
                print(name, torch.abs(pipe_grad - orig_grad) / orig_grad)
                print(name, torch.max(torch.abs(pipe_grad - orig_grad) / orig_grad))

    assert len(not_found_mappings) == 0, f'No qualname mapping found between pipelined and original ' \
                                         f'model: {not_found_mappings}'

    assert len(not_close_orig_grads) == 0, f'Grads not close between pipelined and original ' \
                                           f'model: {not_close_orig_grads}'

    print('correctness checks with original module passed')

    # # # Profiling runs
    # with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
    #     pipe_driver._debug_mask_minibatches = False
    #     pipe_driver.chunks = CHUNKS
    #     out = pipe_driver(ec_input, target)
    #     ref_out = ec_pipe.split_gm(ec_input, target)
    #     print(f'profiling run completed {torch.sum(ref_out)} ref {torch.sum(ref_out)}')
    # if PROFILING_ENABLED:
    #     prof.export_chrome_trace(f'{os.path.splitext(os.path.basename(__file__))[0]}.json')


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
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    # Interleaved 1F1B uses less ranks than number of stages
    if args.schedule == 'Interleaved1F1B':
        args.world_size = 2

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
