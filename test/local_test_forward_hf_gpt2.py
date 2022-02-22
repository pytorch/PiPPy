import inspect

import torch
import torch.distributed.rpc as rpc

import transformers.utils.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain
from transformers import *

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

import os

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

if local_rank == 0:
    bs = 20
    seq_length = 512

    REPLICATE = os.environ.get('REPLICATE', '0') != '0'
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if REPLICATE else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {REPLICATE} -> {MULTI_USE_PARAM_CONFIG}')

    gpt2 = GPT2Model(GPT2Config())
    gpt2_input = torch.zeros(bs, seq_length, dtype=torch.long).random_(gpt2.config.vocab_size)
    gpt2(gpt2_input)

    for i in range(gpt2.config.n_layer):
        annotate_split_points(gpt2, {f'h.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(gpt2, {'ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})

    input_names = gpt2.dummy_inputs.keys()
    sig = inspect.signature(gpt2.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    hf_tracer = fx.HFTracer()

    gpt2_pipe = Pipe.from_tracing(gpt2, MULTI_USE_PARAM_CONFIG, tracer=hf_tracer, concrete_args=concrete_args)

    optimizer = torch.optim.SGD(gpt2_pipe.parameters(), 0.01)

    pipe_driver = PipelineDriverFillDrain(gpt2_pipe, world_size)

    # # Warm up and correctness runs
    out = pipe_driver.run(gpt2_input, chunks=5, _debug_mask_minibatches=True)
    ref_out = gpt2_pipe(gpt2_input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_allclose(out, ref_out)
        print(f'equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}')

    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        out = pipe_driver.run(gpt2_input, chunks=5, _debug_mask_minibatches=False)
        ref_out = gpt2_pipe(gpt2_input)
        print(f'profiling run completed {torch.sum(ref_out)} ref {torch.sum(ref_out)}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace('pipe.csv')

rpc.shutdown()
