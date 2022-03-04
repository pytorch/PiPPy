import inspect

import torch
import torch.distributed.rpc as rpc

import transformers.utils.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points
from pippy.microbatch import TensorChunkSpec
from pippy.PipelineDriver import PipelineDriverFillDrain
from transformers import *

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

import os

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

@torch.fx.wrap
def torch_ones_wrapper(*args, **kwargs):
    return torch.ones(*args, **kwargs)

if local_rank == 0:
    bs = 20
    seq_length = 512

    REPLICATE = os.environ.get('REPLICATE', '0') != '0'
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if REPLICATE else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {REPLICATE} -> {MULTI_USE_PARAM_CONFIG}')

    bert = BertModel(BertConfig())
    bert.eval()
    bert_input = torch.zeros(bs, seq_length, dtype=torch.long).random_(bert.config.vocab_size)
    bert(bert_input)

    for i in range(bert.config.num_hidden_layers):
        annotate_split_points(bert, {f'encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(bert, {'pooler': PipeSplitWrapper.SplitPoint.BEGINNING})

    input_names = bert.dummy_inputs.keys()
    sig = inspect.signature(bert.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    bert_traced = fx.symbolic_trace(bert)

    # HACK to replace HF's non-resolvable wrappers with the original
    # tensor constructor functions
    # Requires this patch to HF: https://github.com/jamesr66a/transformers/commit/ede9d30f36f12be390692617ef76e2928b1612bd
    for node in bert_traced.graph.nodes:
        if node.op == 'call_function':
            if getattr(node.target, '_orig', None) == torch.ones:
                node.target = torch_ones_wrapper

    bert_traced.recompile()

    bert_pipe = Pipe._from_traced(bert, bert_traced, MULTI_USE_PARAM_CONFIG)

    # bert_pipe = Pipe.from_tracing(bert, MULTI_USE_PARAM_CONFIG, tracer=hf_tracer, concrete_args=concrete_args)

    optimizer = torch.optim.SGD(bert_pipe.parameters(), 0.01)

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec = {}
    output_chunk_spec = {'last_hidden_state' : TensorChunkSpec(0), 'pooler_output': TensorChunkSpec(0)}

    pipe_driver = PipelineDriverFillDrain(bert_pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size)

    # # Warm up and correctness runs
    out = pipe_driver.run((bert_input,), {}, chunks=5, _debug_mask_minibatches=True)
    ref_out = bert_pipe(bert_input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_allclose(out['last_hidden_state'], ref_out['last_hidden_state'])
        torch.testing.assert_allclose(out['pooler_output'], ref_out['pooler_output'])
        print(f'equivalence test passed {torch.sum(out["last_hidden_state"])} ref {torch.sum(ref_out["last_hidden_state"])}')

    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        out = pipe_driver.run((bert_input,), {}, chunks=5, _debug_mask_minibatches=False)
        ref_out = bert_pipe(bert_input)
        print(f'profiling run completed {torch.sum(out["last_hidden_state"])} ref {torch.sum(ref_out["last_hidden_state"])}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace('pipe.csv')

rpc.shutdown()
