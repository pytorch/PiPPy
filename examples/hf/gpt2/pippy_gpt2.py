# Copyright (c) Meta Platforms, Inc. and affiliates
import inspect
import os
import time
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
from pippy.utils import get_argparser

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

pippy.fx.Tracer.proxy_buffer_attributes = True


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_split_points(gpt2, decoders_per_rank):
    for i in range(0, gpt2.config.n_layer // decoders_per_rank):
        annotate_split_points(gpt2, {f'transformer.h.{i * decoders_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(gpt2, {f'transformer.ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})
    return gpt2.config.n_layer // decoders_per_rank + 2


def calc_flop(args, conf):
    # https://arxiv.org/pdf/2104.04473.pdf page 8, formula 3
    B = args.batch_size
    s = args.seq_length
    l = conf.n_layer
    h = conf.n_embd
    V = conf.vocab_size
    return 96 * B * s * l * h * h * (1 + s/6/h + V/16/l/h)


def run_gspmd(pp_ranks, args):
    print(args)
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    assert args.world_size >= 4, "This program requires at least 3 workers + 1 master"

    config = GPT2Config()
    config.n_embd = args.n_embd or config.n_embd
    config.n_layer = args.n_layer or config.n_layer
    config.n_head = args.n_head or config.n_head
    print("GPT-2 model instantiation started")
    start = time.time()
    gpt2 = GPT2LMHeadModel(config)
    finish = time.time()
    print(f"GPT-2 model instantiation finished in {(finish - start) / 60:1.2f} minutes")
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

    all_worker_ranks = pp_ranks[
        pippy.utils.exclude_master : pippy.utils.exclude_master
        + number_of_workers
    ]
    chunks = len(all_worker_ranks)
    seq_length = args.seq_length
    batch_size = args.batch_size * chunks
    vocab_size = gpt2.config.vocab_size

    device = args.device
    print("Using device:", device)

    gpt2_input_dict = {
        'input_ids': torch.empty(batch_size, seq_length, dtype=torch.long, device=device).random_(vocab_size),
        'labels': torch.empty(batch_size, seq_length, dtype=torch.long, device=device).random_(vocab_size),
        'position_ids': torch.arange(0, seq_length, dtype=torch.long, device=device)}

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

    # Materialize model differently depending on run mode
    if args.gspmd == 1:
        print(f"Deferring stage init on device {device}")
        gpt2_pipe.defer_stage_init(device)
        # Make sure every rank has deferred its stage init before master creates the driver
        pippy.utils.pp_group_barrier()
    else:
        gpt2_pipe.to(device)

    if args.rank != 0:
        # Workers return here
        return

    # gpt2_pipe(**gpt2_input_dict)

    for i, sm in enumerate(gpt2_pipe.split_gm.children()):
        print(f"submod_{i} {get_number_of_params(sm) // 10 ** 6}M params")

    args_chunk_spec = ()
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'labels': TensorChunkSpec(0), 'position_ids': None}
    output_chunk_spec = {'loss': CustomReducer(torch.tensor(0.0), lambda a, b: a + b), 'logits': TensorChunkSpec(0),
                         'past_key_values': [[TensorChunkSpec(0) for _ in range(2)] for _ in range(config.n_layer)]}
    pipe_driver: PipelineDriverBase = schedules[args.schedule](gpt2_pipe, chunks, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec, len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                               _debug_mask_minibatches=False,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint),
                                                              )

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    if args.warmup_batches > 0:
        print(f'Running {args.warmup_batches} warm-up batches')
        for i in range(args.warmup_batches):
            pipe_driver(**gpt2_input_dict)

    FLOP = calc_flop(args, config)
    print(f"FLOP per iteration {FLOP}")
    print(f'Running GPT-2 pipeline {args.batches} batches for TFLOP/s/GPU measurement.')

    start = time.time()
    for i in range(args.batches):
        pipe_driver(**gpt2_input_dict)
    finish = time.time()
    total_latency = finish - start
    print(f"TFLOP/s/GPU: {FLOP/1e12/total_latency}")

    print(f'Running GPT-2 pipeline {args.batches} batches for visualization.')
    batches_events_contexts = []
    for i in range(args.batches):
        pipe_driver(**gpt2_input_dict)
        batches_events_contexts.append(pipe_driver.retrieve_events())
    all_events_contexts: EventsContext = reduce(lambda c1, c2: EventsContext().update(c1).update(c2),
                                                batches_events_contexts, EventsContext())
    pipe_visualized_filename = f"{this_file_name}_visualized.json"
    with open(pipe_visualized_filename, "w") as f:
        f.write(events_to_json(all_events_contexts))
    print(f"Saved {pipe_visualized_filename}")
    print('Finished.')


if __name__ == "__main__":
    parser = get_argparser(default_schedule=schedules.keys(), default_world_size=16)
    parser.add_argument('--rpc_timeout', type=int, default=1800)
    parser.add_argument('--num_worker_threads', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--warmup_batches', type=int, default=0)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=16)

    parser.add_argument('--n_embd', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)

    parser.add_argument('--gspmd', type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    run_pippy(run_gspmd, args)
