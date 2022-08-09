# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
import socket
from typing import Dict

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import transformers.utils.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverBase
from pippy.microbatch import TensorChunkSpec
import pippy.fx
from test_commons import tp_transports # type: ignore
from transformers import GPT2Model, GPT2Config

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
}

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

@pippy.fx.wrap
def torch_arange_wrapper(*args, **kwargs):
    return torch.arange(*args, **kwargs)


class HFGPT2Tracer(fx.HFTracer):
    def trace(self, root, concrete_args=None):
        graph = super().trace(root, concrete_args)
        # HACK to replace HF's non-resolvable wrappers with the original
        # tensor constructor functions
        # Requires this patch to HF: https://github.com/jamesr66a/transformers/commit/ede9d30f36f12be390692617ef76e2928b1612bd
        for node in graph.nodes:
            if node.op == 'call_function':
                if getattr(node.target, '_orig', None) == torch.arange:
                    node.target = torch_arange_wrapper
        return graph


pippy.fx.Tracer.proxy_buffer_attributes = True


def run_master(args):
    bs = 20
    seq_length = 32

    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')

    print("Using schedule:", args.schedule)

    gpt2 = GPT2Model(GPT2Config(use_cache=False))
    gpt2.to(args.device)
    gpt2.eval()
    gpt2_input = torch.zeros(bs, seq_length, dtype=torch.long,
            device=args.device).random_(gpt2.config.vocab_size)

    for i in range(gpt2.config.n_layer):
        annotate_split_points(gpt2, {f'h.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(gpt2, {'ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})

    input_names = gpt2.dummy_inputs.keys()
    sig = inspect.signature(gpt2.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    hf_tracer = HFGPT2Tracer()

    print('Instantiating GPT2 Pipeline')
    gpt2_pipe = Pipe.from_tracing(gpt2, MULTI_USE_PARAM_CONFIG, tracer=hf_tracer, concrete_args=concrete_args)

    assert gpt2.config.n_layer + 2 == len(list(gpt2_pipe.split_gm.children()))

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec: Dict = {}
    output_chunk_spec = {'last_hidden_state': TensorChunkSpec(0)}

    pipe_driver: PipelineDriverBase = schedules[args.schedule](gpt2_pipe, 5, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec,
                                                               args.world_size,
                                                               _debug_mask_minibatches=True,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint))

    # # Warm up and correctness runs
    print('Running GPT2 pipeline. NB: if this is too slow, set OMP_NUM_THREADS to a higher value')
    out = pipe_driver(gpt2_input)
    print('Running reference pipeline')
    ref_out = gpt2_pipe(gpt2_input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_close(out['last_hidden_state'], ref_out['last_hidden_state'])
        print(
            f'equivalence test passed {torch.sum(out["last_hidden_state"])} ref {torch.sum(ref_out["last_hidden_state"])}')

    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        pipe_driver._debug_mask_minibatches=False
        pipe_driver.chunks = 5
        out = pipe_driver(gpt2_input)
        ref_out = gpt2_pipe(gpt2_input)
        print(
            f'profiling run completed {torch.sum(out["last_hidden_state"])} ref {torch.sum(ref_out["last_hidden_state"])}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace(f'{os.path.splitext(os.path.basename(__file__))[0]}.json')


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
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 14)))
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
