# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import torch
import pippy
import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer, inject_pipeline_forward
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed._tensor import (
    DeviceMesh,
)
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )

from tp_utils import parallelize_MLP_block, parallelize_stage_llama_MLP_block, find_mlp_layers, find_mlp_layers_pattern, tp_stage, even_cut
from utils import print_submodules
import time 

pippy.fx.Tracer.proxy_buffer_attributes = True


def run_all(args):
    device_type = "cuda" if args.cuda else "cpu"
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    model = args.model
    
    model.eval()
    model.config.use_cache = False  # don't output `past_key_values`
    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
        torch.cuda.set_device(dev_id)
    model.to(args.device)
    ranks = torch.arange(args.world_size)
    rank_mesh = ranks.reshape(args.pp_group_size, args.tp_group_size)
    pp_dim = 0
    tp_dim = 1
    
    dm = DeviceMesh(
        device_type,
        rank_mesh,
    )

    print(f"device mesh {dm}")
    print("====================================")
    
    pp_group = dm.get_dim_groups()[pp_dim]
    # Figure out my PP and TP rank
    pp_rank = args.rank // args.tp_group_size
    tp_rank = args.rank % args.tp_group_size
    print(f"Global rank {args.rank}, pp rank: {pp_rank}, tp rank: {tp_rank}")
    
    num_ranks = args.pp_group_size
    
    # make some sample input to be used for concrete_args for HF tracer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompt = "Hey, are you conscious? Can you talk to me?"
    input = tokenizer(prompt, return_tensors="pt")
    
    input_ids = input["input_ids"]
    
    # logging the model modules, this is helpful for understanding/ eyeballing 
    # which layers are in the model/ candidates for TP.
    if rank==0:
        print_submodules(model)
    
   # TODO automating the MLP layer detection for TP 
    # mlp_layers = find_mlp_layers(model)
    # mlp_layer_names = find_mlp_layers_pattern(mlp_layers)
    
  
    # auto split policy is not working for 
    # split_policy = pippy.split_into_equal_size(args.pp_group_size)
    input_ids = input_ids.to(args.device)
    
    # Use default value for kwargs other than `input_ids`
    concrete_args = pippy.create_default_args(
        model,
        except_keys="input_ids",
    )
    if 'bloom' in args.model_name:
        # Used to avoid a control flow and tracing `len` call in BloomForCausalLM that looks like this:
        # `if len(deprecated_arguments) > 0:`
        concrete_args.setdefault("deprecated_arguments", {})
    
    
    # tp(model, dm)
    
    # split strategy of the model for even split of layers on each rank
    even_cut(model, model.config.num_hidden_layers, args.pp_group_size)
    compile_time_start = time.perf_counter()
    # compile_stage to FX trace the model and partition it into multiple stages
    stage= pippy.compile_stage(
        model,
        rank=pp_rank,
        num_ranks= num_ranks,
        num_chunks=args.chunks,
        device=args.device,
        example_inputs=[input_ids],
        group=pp_group,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
    )
    compile_time_end = time.perf_counter()-compile_time_start
    print(f"compile time took {compile_time_end}s")
    print("==============================================")
    TP_time_start = time.perf_counter()
    parallelize_stage_llama_MLP_block(stage, model.config.num_hidden_layers, args.pp_group_size,pp_rank, dm)
    TP_time_end = time.perf_counter()-TP_time_start
    print(f"TP time took {TP_time_end}s")
    print("==============================================")
    
   
    
    if pp_rank == 0:
        out = stage(input_ids)
    elif pp_rank == args.pp_group_size - 1:
        out = stage()
    else:
        out= stage()
    
    
    dist.barrier()
    print(f"Rank {args.rank} completes")
    # Master continues
    # print_mem_usage()
    # Inject pipeline driver's forward function back to original model to support HF's `generate()` method
    # inject_pipeline_forward(model, pipe_driver)
    # Generate text based on prompt
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # prompt = "Hey, are you conscious? Can you talk to me?"
    # input = tokenizer(prompt, return_tensors="pt")
    # input_ids = input["input_ids"].to(args.device)
    # generate_time_start = time.perf_counter()
    # outputs = model.generate(input_ids, max_new_tokens=30)
    # generate_time_end = time.perf_counter()-generate_time_start
    # print(f"Generate time took {generate_time_end}s")
    # print("==============================================")
    # response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(response)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 8)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--chunks', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--tp_group_size', type=int, default=2)
    parser.add_argument('--pp_group_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--dtype', type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument('--index_filename', type=str, default=None, help="The director of model's index.json file")
    parser.add_argument('--checkpoint_prefix', type=str, default=None, help="Prefix to add to the weight names in checkpoint map back to model structure")
    args = parser.parse_args()
    assert args.world_size % args.pp_group_size == 0
    args.tp_group_size = args.world_size // args.pp_group_size
    supported_model_categories = ["opt", "gpt", "bloom", "codegen", "llama"]
    # For example:
    # "facebook/opt-350m"
    # "gpt2"
    # "bigscience/bloom-3b"
    # "EleutherAI/gpt-neo-2.7B"
    # "Salesforce/codegen-2B-multi"
    # "cerebras/Cerebras-GPT-13B"
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        # Using float32 as default dtype to correspond to the default "fp32"
        # value for "--dtype"
        print(
            f"Unsupported data type {args.dtype}, "
            "please submit a PR to support it. Falling back to fp32 now."
        )
        dtype = torch.float32
    # Main process loads model
    if any([m in args.model_name
            # Some model names use upper case
            or m.upper() in args.model_name
            for m in supported_model_categories]):
        print(f"Loading model {args.model_name}")
        if args.index_filename is not None:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_pretrained(args.model_name, use_cache=False, torch_dtype=dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, use_cache=False, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")
    args.model = model
    # args.gspmd = 1
     # Init process group
    backend = "nccl" if args.cuda else "gloo"
    args.pg = dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )
    # run_pippy(run_all, args)
    run_all(args)
