# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
from functools import reduce
from typing import Dict
import time

import torch
from transformers import T5ForConditionalGeneration, T5Config

import pippy.fx
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.hf import PiPPyHFTracer
from pippy.microbatch import CustomReducer, TensorChunkSpec
from pippy.visualizer import events_to_json
from pippy import split_on_size_threshold, split_into_equal_size
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from split_utils import add_split_points, _add_split_points

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True
gigabyte_size = 1073741824
megabyte_size = 1048576

def cleanup():
    dist.destroy_process_group()

def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


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


def resolve_pg_per_stage(pp_rank):
    assert pippy.utils.dp_pg_per_pp_rank
    return pippy.utils.dp_pg_per_pp_rank[pp_rank + pippy.utils.exclude_master]


def run_master(pp_ranks, args):

    logger = setup_logger()

    torch.manual_seed(42)

    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)

   

    device = args.device

    t5 = AutoModelForSeq2SeqLM.from_pretrained('t5-11b', use_cache=False)
    t5_config = t5.config
    t5_config.num_layers = args.num_encoder_layers or t5_config.num_layers
    t5_config.num_decoder_layers = args.num_decoder_layers or t5_config.num_decoder_layers
    t5_config.use_cache = False  # don't output `past_key_values`
    t5 = T5ForConditionalGeneration(t5_config)
    t5.eval()
    memory_reserved = format_to_gb(torch.cuda.memory_reserved())
    memory_allocated = format_to_gb(torch.cuda.memory_allocated())
    print("***********************************************")
    print("memory_reserved after model intializaed from HF pretrained", memory_reserved)
    print("memory_allocated after model intializaed from HF pretrained", memory_allocated)
    print("***********************************************")
    print(t5.config)
    print(f"T5 total number of params = {get_number_of_params(t5) // 10 ** 6}M")

    number_of_workers = len(pp_ranks) - pippy.utils.exclude_master
    print(f"number_of_workers = {number_of_workers}")
    # add_split_points(t5, number_of_workers)

    if args.auto_split == "threshold":
        split_policy = split_on_size_threshold(490 * 1e6)
    elif args.auto_split == "equal_size":
        split_policy = split_into_equal_size(number_of_workers)

    all_worker_ranks = pp_ranks[pippy.utils.exclude_master:pippy.utils.exclude_master + number_of_workers]
    chunks = args.chunks or len(all_worker_ranks)
    bs = args.batch_size * chunks
    seq_length = args.seq_length

    print("Using device:", device)

    torch.manual_seed(args.rank)
    inp = torch.empty(bs, seq_length, dtype=torch.long, device=device).random_(t5.config.vocab_size)

    t5_input_dict = {'input_ids': inp, 'decoder_input_ids': inp}


    input_names = t5_input_dict.keys()
    sig = inspect.signature(t5.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    print('Instantiating T5 Pipeline')
    model_init_start = time.time()
    t5_pipe = Pipe.from_tracing(t5, MULTI_USE_PARAM_CONFIG, tracer=PiPPyHFTracer(), concrete_args=concrete_args,
                                output_loss_value_spec=None, split_policy=split_policy
                                )
   
    t5_pipe.defer_stage_init(args.device)

    torch.distributed.barrier(args.pp_group)

    if args.rank!=0:
        return 
    
    split_gm_children = list(t5_pipe.split_gm.children())

    total_params = 0
    for i, sm in enumerate(t5_pipe.split_gm.children()):
        params = get_number_of_params(sm)
        print(f"submod_{i} {params // 10 ** 6}M params")
        total_params += params
    print(f"total {total_params // 10 ** 6}M params")

    args_chunk_spec = ()

    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'decoder_input_ids': TensorChunkSpec(0)}

    output_chunk_spec = {"logits": TensorChunkSpec(0),"encoder_last_hidden_state": TensorChunkSpec(0)}

    pipe_driver: PipelineDriverBase = schedules[args.schedule](t5_pipe, chunks, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec,
                                                               world_size=len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                               _debug_mask_minibatches=False,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint),
                                                                # device=intermediate_device
                                                                )
    model_init_end = time.time()

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print('Running T5 pipeline.')
    for i in range(args.num_batches):
        pipe_driver(**t5_input_dict)

    
    print('Inference is finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 8)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    model_config = os.path.dirname(os.path.realpath(__file__)) + "/" + "t5_200m_config.json"
    parser.add_argument('--model_config', type=str, default=model_config)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=16)
    parser.add_argument('--avg_seqlen', type=int, default=16)
    parser.add_argument('--max_seqlen', type=int, default=16)
    parser.add_argument('--seqlen-stdev', type=int, default=10)

    parser.add_argument('--num_encoder_layers', type=int, default=None)
    parser.add_argument('--num_decoder_layers', type=int, default=None)

    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--visualize', type=int, default=1, choices=[0, 1])
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=1, choices=[0, 1])
    parser.add_argument('--pp_group_size', type=int, default=8)
    parser.add_argument('--exclude_master', type=int, default=0, choices=[0, 1])
    parser.add_argument('--auto_split', type=str, default="equal_size")

    args = parser.parse_args()

    assert args.world_size % args.pp_group_size == 0
   
    args.dp_group_size = args.world_size // args.pp_group_size
    args.gspmd = 1

    run_pippy(run_master, args)
