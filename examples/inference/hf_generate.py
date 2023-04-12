# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

import torch
import pippy
import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer, inject_pipeline_forward
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights


pippy.fx.Tracer.proxy_buffer_attributes = True

gigabyte_size = 1024 ** 3


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

    # Use default value for kwargs other than `input_ids`
    concrete_args = pippy.create_default_args(
        model,
        except_keys="input_ids",
    )
    if 'bloom' in args.model_name:
        # Used to avoid a control flow and tracing `len` call in BloomForCausalLM that looks like this:
        # `if len(deprecated_arguments) > 0:`
        concrete_args.setdefault("deprecated_arguments", {})

    pipe_driver, stage_mod = pippy.all_compile(
        model,
        num_ranks,
        args.chunks,
        split_policy=split_policy,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
        index_filename=args.index_filename,
    )

    params = get_number_of_params(stage_mod)
    print(f"submod_{args.rank} {params // 10 ** 6}M params")

    if args.rank != 0:
        return

    # Master continues
    print_mem_usage()

    # Inject pipeline driver's forward function back to original model to support HF's `generate()` method
    inject_pipeline_forward(model, pipe_driver)

    # Generate text based on prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompt = "Hey, are you conscious? Can you talk to me?"
    input = tokenizer(prompt, return_tensors="pt")
    input_ids = input["input_ids"].to(args.device)
    outputs = model.generate(input_ids, max_length=30)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--chunks', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--pp_group_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--dtype', type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument('--index_filename', type=str, default=None, help="The director of model's index.json file")

    args = parser.parse_args()

    assert args.world_size % args.pp_group_size == 0

    supported_model_categories = ["opt", "gpt2", "bloom", "EleutherAI/gpt", "codegen"]
    # For example:
    # "facebook/opt-350m"
    # "gpt2"
    # "bigscience/bloom-3b"
    #EleutherAI/gpt-neo-2.7B
    #Salesforce/codegen-2B-multi

    # Main process loads model
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    if any([m in args.model_name for m in supported_model_categories]):
        print(f"Loading model {args.model_name}")
        if args.index_filename is not None:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(args.model_name, use_cache=False, torch_dtype=dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, use_cache=False, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    args.model = model
    args.gspmd = 1
    run_pippy(run_all, args)
