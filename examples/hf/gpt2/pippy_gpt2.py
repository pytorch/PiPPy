# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
from functools import reduce

import torch
from transformers import GPT2LMHeadModel, GPT2Config

import pippy.fx
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.hf import PiPPyHFTracer
from pippy.microbatch import CustomReducer, TensorChunkSpec
from pippy.visualizer import events_to_json

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

pippy.fx.Tracer.proxy_buffer_attributes = True


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# TODO: Fails! Why??? https://gist.github.com/pbelevich/f4e78c6ed2fdabc8b02ab15e254935fd
# def add_split_points(gpt2, layers_per_rank):
#     for i in range(0, gpt2.config.n_layer // layers_per_rank):
#         annotate_split_points(gpt2, {f'transformer.h.{i * layers_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
#     annotate_split_points(gpt2, {
#         f'transformer.h.{gpt2.config.n_layer // layers_per_rank - 1}': PipeSplitWrapper.SplitPoint.END})
#     return gpt2.config.n_layer // layers_per_rank + 2


# TODO: Fails! Why??? https://gist.github.com/pbelevich/f4e78c6ed2fdabc8b02ab15e254935fd
# def add_split_points(gpt2, layers_per_rank):
#     for i in range(0, gpt2.config.n_layer // layers_per_rank):
#         annotate_split_points(gpt2, {f'transformer.h.{i * layers_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
#     annotate_split_points(gpt2, {f'lm_head': PipeSplitWrapper.SplitPoint.BEGINNING})
#     return gpt2.config.n_layer // layers_per_rank + 2


# TODO: Fails! Why??? https://gist.github.com/pbelevich/f4e78c6ed2fdabc8b02ab15e254935fd
# def add_split_points(gpt2, decoders_per_rank):
#     annotate_split_points(gpt2, {f'transformer.drop': PipeSplitWrapper.SplitPoint.END})
#     for i in range(0, gpt2.config.n_layer // decoders_per_rank):
#         annotate_split_points(gpt2, {
#             f'transformer.h.{i * decoders_per_rank + decoders_per_rank - 1}': PipeSplitWrapper.SplitPoint.END})
#     return gpt2.config.n_layer // decoders_per_rank + 2


def add_split_points(gpt2, decoders_per_rank):
    for i in range(0, gpt2.config.n_layer // decoders_per_rank):
        annotate_split_points(gpt2, {f'transformer.h.{i * decoders_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(gpt2, {f'transformer.ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})
    return gpt2.config.n_layer // decoders_per_rank + 2


def run_master(_, args):
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)

    assert args.world_size >= 4, "This program requires at least 3 workers + 1 master"

    gpt2 = GPT2LMHeadModel(GPT2Config())
    gpt2.eval()
    print(gpt2.config)
    print(f"GPT-2 total number of params = {get_number_of_params(gpt2) // 10 ** 6}M")

    emb_head = 2  # embeddings + head
    master_emb_head = 1 + emb_head  # master + embeddings + head
    decoders_per_rank = (gpt2.config.n_layer + (args.world_size - master_emb_head) - 1) // (
            args.world_size - master_emb_head)  # a divider of gpt2.config.n_layer: [1, 2, 3, 4, 6, 12]
    print(f"decoders_per_rank = {decoders_per_rank}")
    number_of_workers = emb_head + gpt2.config.n_layer // decoders_per_rank  # 3 + a divider of gpt2.config.n_layer: [4, 5, 6, 7, 9, 15]
    print(f"number_of_workers = {number_of_workers}")

    all_worker_ranks = list(range(1, 1 + number_of_workers))  # exclude master rank = 0
    chunks = len(all_worker_ranks)
    batches = 1
    bs = 1 * chunks
    seq_length = 16

    device = args.device
    print("Using device:", device)

    gpt2_input_dict = {
        'input_ids': torch.empty(bs, seq_length, dtype=torch.long, device=device).random_(gpt2.config.vocab_size),
        'labels': torch.empty(bs, seq_length, dtype=torch.long, device=device).random_(gpt2.config.vocab_size),
        'position_ids': torch.arange(0, seq_length, dtype=torch.long,
                                     device=device)}  # needed because otherwise it is instantiated on cpu

    sm_cnt = add_split_points(gpt2, decoders_per_rank)
    assert sm_cnt == len(all_worker_ranks), f"sm_cnt = {sm_cnt} all_worker_ranks = {all_worker_ranks}"

    # print(gpt2)

    input_names = gpt2_input_dict.keys()
    sig = inspect.signature(gpt2.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    print('Instantiating GPT-2 Pipeline')
    output_loss_value_spec = {'loss': True, 'logits': False,
                              'past_key_values': [[False for _ in range(2)] for _ in range(12)]}
    gpt2_pipe = Pipe.from_tracing(gpt2, MULTI_USE_PARAM_CONFIG, tracer=PiPPyHFTracer(), concrete_args=concrete_args,
                                  output_loss_value_spec=output_loss_value_spec, deep_copy_module=False)
    assert sm_cnt == len(list(gpt2_pipe.split_gm.children()))
    gpt2_pipe.to(device)

    # gpt2_pipe(**gpt2_input_dict)

    for i, sm in enumerate(gpt2_pipe.split_gm.children()):
        print(f"submod_{i} {get_number_of_params(sm) // 10 ** 6}M params")

    args_chunk_spec = ()
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'labels': TensorChunkSpec(0), 'position_ids': None}
    output_chunk_spec = {'loss': CustomReducer(torch.tensor(0.0), lambda a, b: a + b), 'logits': TensorChunkSpec(0),
                         'past_key_values': [[TensorChunkSpec(0) for _ in range(2)] for _ in range(12)]}
    pipe_driver: PipelineDriverBase = schedules[args.schedule](gpt2_pipe, chunks, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec, len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                               _debug_mask_minibatches=False,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint))

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print('Running GPT2 pipeline.')
    pipe_visualized_filename = f"{this_file_name}_visualized.json"
    batches_events_contexts = []
    for i in range(batches):
        pipe_driver(**gpt2_input_dict)
        batches_events_contexts.append(pipe_driver.retrieve_events())

    all_events_contexts: EventsContext = reduce(lambda c1, c2: EventsContext().update(c1).update(c2),
                                                batches_events_contexts, EventsContext())
    with open(pipe_visualized_filename, "w") as f:
        f.write(events_to_json(all_events_contexts))
    print(f"Saved {pipe_visualized_filename}")
    print('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 16)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    run_pippy(run_master, args)
