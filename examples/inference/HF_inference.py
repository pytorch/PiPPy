# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
import time

import torch
import pippy.fx
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.hf import PiPPyHFTracer
from pippy import split_on_size_threshold, split_into_equal_size
from transformers import  AutoModelForSeq2SeqLM
from transformers import OPTModel, BloomModel
from PIL import Image
import requests
from transformers import AutoFeatureExtractor, RegNetModel 


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



def run_master(pp_ranks, args):

    # logger = setup_logger()

    torch.manual_seed(42)

    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)
    
    device = args.device
    model = args.model
    if 'regnet' in args.model_name:
        feature_extractor = args.feature_extractor

    model_config = model.config

    model_config.use_cache = False  # don't output `past_key_values`
    model.eval()
    print(model.config)
    print(f"model total number of params = {get_number_of_params(model) // 10 ** 6}M")

    number_of_workers = len(pp_ranks) - pippy.utils.exclude_master
    print(f"number_of_workers = {number_of_workers}")

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

    # preparing inputs based on the model choice
    if 't5' in args.model_name:
        inp = torch.empty(bs, seq_length, dtype=torch.long, device=device).random_(model.config.vocab_size)
        model_input_dict = {'input_ids': inp, 'decoder_input_ids': inp}
    elif 'opt' or 'bloom' in args.model_name:
        inp = torch.empty(bs, seq_length, dtype=torch.long, device=device).random_(model.config.vocab_size)
        model_input_dict = {'input_ids': inp}

    elif 'regnet' in args.model_name:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = feature_extractor(image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        model_input_dict = {'pixel_values': inputs["pixel_values"]}

    input_names = model_input_dict.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    print('Instantiating model Pipeline')
    model_init_start = time.time()
    model_pipe = Pipe.from_tracing(model, MULTI_USE_PARAM_CONFIG, tracer=PiPPyHFTracer(), concrete_args=concrete_args,
                                output_loss_value_spec=None, split_policy=split_policy
                                )
   
    model_pipe.defer_stage_init(args.device)

    pippy.utils.pp_group_barrier()

    if args.rank!=0:
        return 
    
    split_gm_children = list(model_pipe.split_gm.children())

    total_params = 0
    for i, sm in enumerate(model_pipe.split_gm.children()):
        params = get_number_of_params(sm)
        print(f"submod_{i} {params // 10 ** 6}M params")
        total_params += params
    print(f"total {total_params // 10 ** 6}M params")

    pipe_driver: PipelineDriverBase = schedules[args.schedule](model_pipe, chunks,
                                                               world_size=len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                                )
    model_init_end = time.time()

    print("Model initialization time")
    print("=========================")
    print("{} seconds".format(model_init_end - model_init_start))
 
    memory_reserved = format_to_gb(torch.cuda.memory_reserved())
    memory_allocated = format_to_gb(torch.cuda.memory_allocated())
    print("memory_reserved after model intializaed with pipelines on each rank")
    print("===================================================================")
    print(" {} GB".format(memory_reserved))
    print("memory_allocated after model intializaed with pipelines on each rank")
    print("===================================================================")
    print(" {} GB".format(memory_allocated))


    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print('Running model pipeline.')
    for i in range(args.num_batches):
        pipe_driver(**model_input_dict)

    print('Inference is finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    parser.add_argument('--model_name', type=str, default='t5-3b')
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
    parser.add_argument('--pp_group_size', type=int, default=4)
    parser.add_argument('--exclude_master', type=int, default=0, choices=[0, 1])
    parser.add_argument('--auto_split', type=str, default="equal_size")


    args = parser.parse_args()

    assert args.world_size % args.pp_group_size == 0
    args.dp_group_size = args.world_size // args.pp_group_size
    args.gspmd = 1
    if 't5' in args.model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, use_cache=False)
    if 'opt' in args.model_name:
        model = OPTModel.from_pretrained(args.model_name, use_cache=False)
    if 'bloom' in args.model_name:
        model = BloomModel.from_pretrained(args.model_name, use_cache=False)
    if 'regnet' in args.model_name:
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/regnet-y-10b-seer")
        model = RegNetModel.from_pretrained("facebook/regnet-y-10b-seer")
        args.feature_extractor = feature_extractor
    args.model = model
    run_pippy(run_master, args)
