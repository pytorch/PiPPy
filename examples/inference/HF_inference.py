# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import os
import time

import torch
import pippy
import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer
from pippy import split_on_size_threshold, split_into_equal_size
from transformers import  AutoModelForSeq2SeqLM
from transformers import OPTModel, BloomModel
from PIL import Image
import requests
from transformers import AutoFeatureExtractor, RegNetModel 


pippy.fx.Tracer.proxy_buffer_attributes = True

gigabyte_size = 1024 ** 3
megabyte_size = 1024 ** 2


def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


def print_mem_usage():
    memory_reserved = format_to_gb(torch.cuda.memory_reserved())
    memory_allocated = format_to_gb(torch.cuda.memory_allocated())
    print(
        f"memory_reserved: {memory_reserved} GB, "
        f"memory_allocated: {memory_allocated} GB"
    )


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_input(args):
    bs = args.batch_size * args.chunks
    seq_length = args.seq_length
    model_config = args.model.config
    torch.manual_seed(args.rank)

    # preparing inputs based on the model choice
    if 't5' in args.model_name:
        inp = torch.empty(bs, seq_length, dtype=torch.long, device=args.device).random_(model_config.vocab_size)
        model_input_dict = {'input_ids': inp, 'decoder_input_ids': inp}
    elif 'opt' or 'bloom' in args.model_name:
        inp = torch.empty(bs, seq_length, dtype=torch.long, device=args.device).random_(model_config.vocab_size)
        model_input_dict = {'input_ids': inp}
    elif 'regnet' in args.model_name:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = args.feature_extractor(image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"]
        model_input_dict = {'pixel_values': inputs["pixel_values"]}

    return model_input_dict


def run_all(pp_ranks, args):
    model = args.model
    model.eval()
    model.config.use_cache = False  # don't output `past_key_values`
    num_ranks = len(pp_ranks)

    if args.rank == 0:
        print("Using schedule:", args.schedule)
        print(model.config)
        print(f"model total number of params = {get_number_of_params(model) // 10 ** 6}M")

    if args.auto_split == "threshold":
        split_policy = split_on_size_threshold(490 * 1e6)
    elif args.auto_split == "equal_size":
        split_policy = split_into_equal_size(num_ranks)

    model_input_dict = generate_input(args)
    input_names = model_input_dict.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    model_init_start = time.time()

    pipe_driver, stage_mod = pippy.all_compile(
        model,
        num_ranks,
        args.chunks,
        schedule=args.schedule,
        split_policy=split_policy,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
    )

    model_init_end = time.time()

    if args.rank == 0:
        print(f"Model init time: {model_init_end - model_init_start} s")
        print('Running model pipeline.')

        for _ in range(args.num_batches):
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
    parser.add_argument('--chunks', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=16)
    parser.add_argument('--avg_seqlen', type=int, default=16)
    parser.add_argument('--max_seqlen', type=int, default=16)
    parser.add_argument('--seqlen-stdev', type=int, default=10)

    parser.add_argument('-s', '--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--visualize', type=int, default=1, choices=[0, 1])
    parser.add_argument('--pp_group_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--auto_split', type=str, default="equal_size")

    args = parser.parse_args()

    assert args.world_size % args.pp_group_size == 0

    # Main process loads model
    print(f"Loading model {args.model_name}")
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

    args.gspmd = 1
    run_pippy(run_all, args)
