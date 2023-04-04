# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import time

import torch
import pippy
import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer, inject_pipeline_forward
from pippy import split_on_size_threshold, split_into_equal_size
from transformers import  AutoModelForCausalLM, AutoTokenizer
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



def run_all(pp_ranks, args):
    model = args.model
    model.eval()
    model.config.use_cache = False  # don't output `past_key_values`
    num_ranks = len(pp_ranks)

    if args.rank == 0:
        print(model.config)
        print(f"model total number of params = {get_number_of_params(model) // 10 ** 6}M")

    split_policy = pippy.split_into_equal_size(num_ranks)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompt = "Hey, are you consciours? Can you talk to me?"
    input = tokenizer(prompt, return_tensors="pt")

    input= {key: value.to(args.device) for key, value in input.items() if key in ["input_ids"]}

    # Use default value for other kwargs than those in `model_input_dict`
    concrete_args = pippy.create_default_args(
        model,
        except_keys=input.keys()
    )

    pipe_driver, stage_mod = pippy.all_compile(
        model,
        num_ranks,
        args.chunks,
        split_policy=split_policy,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
    )

    params = get_number_of_params(stage_mod)
    print(f"submod_{args.rank} {params // 10 ** 6}M params")

    if args.rank != 0:
        return

    # Master continues
    print_mem_usage()

    # Inject pipeline driver's forward function back to original model to support HF's `generate()` method
    inject_pipeline_forward(model, pipe_driver)

    outputs = model.generate(**input, max_length=30)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--chunks', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=16)
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--pp_group_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))

    args = parser.parse_args()

    assert args.world_size % args.pp_group_size == 0

    # Main process loads model
    print(f"Loading model {args.model_name}")
    if 't5' not in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, use_cache=False)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    args.model = model
    args.gspmd = 1
    run_pippy(run_all, args)
