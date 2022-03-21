# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.distributed.rpc as rpc
import logging
import copy

from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec, CustomReducer, split_args_kwargs_into_chunks

# TODOs for implementing forward/backward/loss with schedules:
# * ability to switch between full-batch loss vs. per-microbatch loss. shen mentioned
# this might change numerics. So we should have the ability to compute loss over
# the whole minibatch rather than doing it for each micro-batch

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

VERBOSE = bool(os.environ.get('VERBOSE', False))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

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

# WAR for SEV remediation https://github.com/pytorch/pytorch/commit/2337d4e5036a87f473decd2b1f6fe0439499902c
torch.fx.Tracer.proxy_buffer_attributes = True

if local_rank == 0:
    torch.manual_seed(42)

    d_hid = 50
    bs = 503
    CHUNKS = 5
    DEBUG_MASK_MINIBATCHES = True
    REF_USE_MICROBATCHES = True
    REPLICATE = os.environ.get('REPLICATE', '0') != '0'
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if REPLICATE else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {REPLICATE} -> {MULTI_USE_PARAM_CONFIG}')

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
    ec(torch.randn(bs, d_hid))
    ec.train()

    # TODO: works with sum, need to define semantics for e.g. mean
    mse_loss = torch.nn.MSELoss(reduction='sum')

    ec_pipe = Pipe.from_tracing(ec, MULTI_USE_PARAM_CONFIG, loss_fn=mse_loss)
    print(ec_pipe.split_gm)

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}
    output_chunk_spec = CustomReducer(torch.tensor(0.0), lambda a, b: a + b)

    pipe_driver = PipelineDriverFillDrain(ec_pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size)

    input = torch.randn(bs, d_hid)
    target = torch.randn(bs, d_hid)

    # TODO: distributed optimizer
    out = pipe_driver.run((input, target), {}, chunks=CHUNKS, _debug_mask_minibatches = DEBUG_MASK_MINIBATCHES)

    all_grad_qualnames = {k: None for k, v in ec_pipe.named_parameters()}

    replicated_params_qualnames = {}

    # Shared parameter sync. TODO: move this to actual runtime
    for param_set in ec_pipe.replicated_params:
        grad_values = []
        for module_name, param_qualname in param_set.items():
            assert module_name in pipe_driver.remote_stage_executor_rrefs
            rank, module_rref = pipe_driver.remote_stage_executor_rrefs[module_name]
            grad_value = rpc.rpc_sync(rank, get_grad_from_executor, (module_rref, param_qualname))
            grad_values.append(grad_value)
            all_grad_qualnames.setdefault(f'split_gm.{module_name}.{param_qualname}')

        synced_value = torch.sum(torch.stack(grad_values), dim=0)

        for module_name, param_qualname in param_set.items():
            assert module_name in pipe_driver.remote_stage_executor_rrefs
            rank, module_rref = pipe_driver.remote_stage_executor_rrefs[module_name]
            rpc.rpc_sync(rank, set_grad_in_executor, (module_rref, param_qualname, synced_value))

            replicated_params_qualnames.setdefault(f'split_gm.{module_name}.{param_qualname}')

    pipe_grads = {}

    for name in all_grad_qualnames:
        assert 'split_gm.' in name
        _, module_name, param_qualname = name.split('.', maxsplit=2)

        assert module_name in pipe_driver.remote_stage_executor_rrefs
        rank, module_rref = pipe_driver.remote_stage_executor_rrefs[module_name]
        grad_value = rpc.rpc_sync(rank, get_grad_from_executor, (module_rref, param_qualname))
        pipe_grads[name] = copy.deepcopy(grad_value)

    optim = torch.optim.SGD(ec_pipe.split_gm.parameters(), lr=0.05)
    optim.zero_grad()
    if REF_USE_MICROBATCHES:
        args_split, kwargs_split = split_args_kwargs_into_chunks((input, target), {}, args_chunk_spec,
                                                            kwargs_chunk_spec, CHUNKS,
                                                            DEBUG_MASK_MINIBATCHES)
        ref_outs = []
        for chunk in range(CHUNKS):
            ref_outs.append(ec_pipe(*args_split[chunk]))
        ref_out = torch.sum(torch.stack(ref_outs))
    else:
        ref_out = ec_pipe(input, target)

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
        torch.testing.assert_allclose(out, ref_out)
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
    orig_loss = mse_loss(ec(input), target)
    orig_loss.backward()
    torch.testing.assert_allclose(out, orig_loss)

    not_close_orig_grads = []
    not_found_mappings = []

    for name in all_grad_qualnames:
        try:
            remapped_qualname = ec_pipe.remap_qualname(name)
        except KeyError:
            not_found_mappings.append(name)
        else:
            orig_grad = ec.get_parameter(remapped_qualname).grad
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
    #     out = pipe_driver.run(input, target, chunks=5, _debug_mask_minibatches = False)
    #     ref_out = ec_pipe.split_gm(input, target)
    #     print(f'profiling run completed {torch.sum(ref_out)} ref {torch.sum(ref_out)}')
    # if PROFILING_ENABLED:
    #     prof.export_chrome_trace('pipe.csv')

rpc.shutdown()
