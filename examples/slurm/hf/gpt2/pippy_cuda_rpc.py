import inspect
import os

import torch
import torch.distributed.rpc as rpc

import transformers.utils.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec
from transformers import *

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

# WAR for SEV remediation https://github.com/pytorch/pytorch/commit/2337d4e5036a87f473decd2b1f6fe0439499902c
torch.fx.Tracer.proxy_buffer_attributes = True


@torch.fx.wrap
def torch_arange_wrapper(*args, **kwargs):
    return torch.arange(*args, **kwargs)


class HFGPT2Tracer(fx.HFTracer):
    def trace(self, root, concrete_args=None, method_names=None):
        graph = super().trace(root, concrete_args, method_names)
        # HACK to replace HF's non-resolvable wrappers with the original
        # tensor constructor functions
        # Requires this patch to HF: https://github.com/jamesr66a/transformers/commit/ede9d30f36f12be390692617ef76e2928b1612bd
        for node in graph.nodes:
            if node.op == 'call_function':
                if getattr(node.target, '_orig', None) == torch.arange:
                    node.target = torch_arange_wrapper
        return graph


def add_split_points(gpt2, layers_per_rank):
    for i in range(gpt2.config.n_layer // layers_per_rank):
        annotate_split_points(gpt2, {f'h.{i * layers_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(gpt2, {'ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})


def run_master():
    REPLICATE = os.environ.get('REPLICATE', '0') != '0'
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if REPLICATE else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {REPLICATE} -> {MULTI_USE_PARAM_CONFIG}')

    bs = 20
    seq_length = 32
    layers_per_rank = 1

    device = torch.device('cuda:0')

    gpt2 = GPT2Model(GPT2Config(use_cache=False))
    gpt2.to(device)
    gpt2.eval()
    gpt2_input = torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(gpt2.config.vocab_size)

    add_split_points(gpt2, layers_per_rank)

    input_names = gpt2.dummy_inputs.keys()
    sig = inspect.signature(gpt2.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    hf_tracer = HFGPT2Tracer()

    print('Instantiating GPT2 Pipeline')
    gpt2_pipe = Pipe.from_tracing(gpt2, MULTI_USE_PARAM_CONFIG, tracer=hf_tracer, concrete_args=concrete_args)
    assert gpt2.config.n_layer // layers_per_rank + 2 == len(list(gpt2_pipe.split_gm.children()))

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec = {}
    output_chunk_spec = {'last_hidden_state': TensorChunkSpec(0)}
    pipe_driver = PipelineDriverFillDrain(gpt2_pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec,
                                          world_size)

    # Warm up and correctness runs
    print('Running GPT2 pipeline. NB: if this is too slow, set OMP_NUM_THREADS to a higher value')
    out = pipe_driver.run((gpt2_input,), {}, chunks=5, _debug_mask_minibatches=True)

    print('Running reference pipeline')
    ref_out = gpt2_pipe(gpt2_input)
    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_allclose(out['last_hidden_state'], ref_out['last_hidden_state'])
        print(
            f'equivalence test passed {torch.sum(out["last_hidden_state"])} ref {torch.sum(ref_out["last_hidden_state"])}')

    # Profiling runs
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        out = pipe_driver.run((gpt2_input,), {}, chunks=5, _debug_mask_minibatches=False)
        ref_out = gpt2_pipe(gpt2_input)
        print(
            f'profiling run completed {torch.sum(out["last_hidden_state"])} ref {torch.sum(ref_out["last_hidden_state"])}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace(f'{os.path.splitext(os.path.basename(__file__))[0]}.json')


if __name__ == '__main__':
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, _transports=["uv"])
    for i in range(world_size):
        options.set_device_map(f"worker{i}", {0: 0})
    rpc.init_rpc(f'worker{rank}', rank=rank, world_size=world_size, rpc_backend_options=options)

    if rank == 0:
        run_master()

    rpc.shutdown()
