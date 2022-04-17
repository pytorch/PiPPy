# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
import socket
from functools import reduce

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import transformers.utils.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, \
    annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.microbatch import CustomReducer, TensorChunkSpec
from pippy.visualizer import events_to_json
from transformers import BertLMHeadModel, BertConfig
from transformers.modeling_utils import ModuleUtilsMixin

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


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.fx.wrap
def torch_ones_wrapper(*args, **kwargs):
    return torch.ones(*args, **kwargs)


@torch.fx.wrap
def torch_create_extended_attention_mask_for_decoder_wrapper(*args, **kwargs):
    return ModuleUtilsMixin.create_extended_attention_mask_for_decoder(*args, **kwargs)


class HFBertTracer(fx.HFTracer):
    def trace(self, root, concrete_args=None, method_names=None):
        graph = super().trace(root, concrete_args, method_names)
        # HACK to replace HF's non-resolvable wrappers with the original
        # tensor constructor functions
        # Requires this patch to HF: https://github.com/jamesr66a/transformers/commit/ede9d30f36f12be390692617ef76e2928b1612bd
        for node in graph.nodes:
            if node.op == 'call_function':
                if getattr(node.target, '_orig', None) == torch.ones:
                    node.target = torch_ones_wrapper
                elif getattr(node.target, '_orig', None) == ModuleUtilsMixin.create_extended_attention_mask_for_decoder:
                    node.target = torch_create_extended_attention_mask_for_decoder_wrapper
        return graph


def add_split_points(bert, layers_per_rank):
    for i in range(1, bert.config.num_hidden_layers // layers_per_rank):
        annotate_split_points(bert,
                              {f'bert.encoder.layer.{i * layers_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    return bert.config.num_hidden_layers // layers_per_rank


def run_master(args):
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)

    all_ranks = list(range(1, args.world_size))  # exclude master rank = 0
    chunks = len(all_ranks)
    batches = 1
    bs = 4 * chunks
    seq_length = 32

    # If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`
    bert = BertLMHeadModel(BertConfig(is_decoder=True))
    bert.eval()
    print(bert.config)
    print(f"BERT total number of params = {get_number_of_params(bert) // 10 ** 6}M")

    bert_input_dict = {'input_ids': torch.zeros(bs, seq_length, dtype=torch.long).random_(bert.config.vocab_size),
                       'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(bert.config.vocab_size)}

    print(f"BERT output keys: {bert(**bert_input_dict).keys()}")

    layers_per_rank = (bert.config.num_hidden_layers + len(all_ranks) - 1) // len(all_ranks)
    add_split_points(bert, layers_per_rank)

    hf_tracer = HFBertTracer()

    input_names = bert_input_dict.keys()
    sig = inspect.signature(bert.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    print('Instantiating BERT Pipeline')
    output_loss_value_spec = {'loss': True, 'logits': False}
    bert_pipe = Pipe.from_tracing(bert, MULTI_USE_PARAM_CONFIG, tracer=hf_tracer, concrete_args=concrete_args,
                                  output_loss_value_spec=output_loss_value_spec)

    for i, sm in enumerate(bert_pipe.split_gm.children()):
        print(f"submod_{i} {get_number_of_params(sm) // 10 ** 6}M params")

    print(f"BERT pipe output keys: {bert_pipe(**bert_input_dict).keys()}")

    args_chunk_spec = ()
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'labels': TensorChunkSpec(0)}
    output_chunk_spec = {'loss': CustomReducer(torch.tensor(0.0), lambda a, b: a + b), 'logits': TensorChunkSpec(0)}
    pipe_driver: PipelineDriverBase = schedules[args.schedule](bert_pipe, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec, args.world_size - 1,
                                                               all_ranks=all_ranks, _debug_mask_minibatches=True)

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print('Running BERT pipeline.')
    pipe_visualized_filename = f"{this_file_name}_visualized.json"
    batches_events_contexts = []
    for i in range(batches):
        pipe_driver.run(chunks, **bert_input_dict)
        batches_events_contexts.append(pipe_driver.retrieve_events())

    # first: save file
    all_events_contexts: EventsContext = reduce(lambda c1, c2: EventsContext().update(c1).update(c2),
                                                batches_events_contexts, EventsContext())
    with open(pipe_visualized_filename, "w") as f:
        f.write(events_to_json(all_events_contexts))
    print(f"Saved {pipe_visualized_filename}")

    # # Warm up and correctness runs
    out = pipe_driver.run(chunks, **bert_input_dict)
    ref_out = bert_pipe(**bert_input_dict)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_close(out['logits'], ref_out['logits'])
        print(
            f'equivalence test passed {torch.sum(out["logits"])} ref {torch.sum(ref_out["logits"])}')

    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        pipe_driver._debug_mask_minibatches = False
        out = pipe_driver.run(chunks, **bert_input_dict)
        ref_out = bert_pipe(**bert_input_dict)
        print(
            f'profiling run completed {torch.sum(out["logits"])} ref {torch.sum(ref_out["logits"])}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace(f'{this_file_name}_profiled.json')


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
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 7)))
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
