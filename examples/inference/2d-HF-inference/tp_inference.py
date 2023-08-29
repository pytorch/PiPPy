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

from utils import print_submodules
import time 
import torch.multiprocessing as mp

def print_submodules(model):
        for name, module in model.named_modules():
            print(f"Module name: {name}")
            # print(module)
            print()
            
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def parallelize_llama_MLP_block(model, module_path, twod_mesh):
    block = model.get_submodule(module_path)
    parallelized_block = parallelize_module(
        module=block,
        device_mesh=twod_mesh,
        parallelize_plan={
            "up_proj": ColwiseParallel(),
            "gate_proj": ColwiseParallel(),
            "down_proj": RowwiseParallel(),
        },
        # tp_mesh_dim=1,
    )
    return parallelized_block

def tp_llama(model, mesh):
    for i in range(model.config.num_hidden_layers):
        print(f" i number of layers {i}*********************")
        block = parallelize_llama_MLP_block(model, f"model.layers.{i}.mlp", mesh)
    

def run_all(args):
    # setup(rank, world_size)
    device_type = "cuda" if args.cuda else "cpu"
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # model = args.model
    
    args.model.eval()
    args.model.config.use_cache = False  # don't output `past_key_values`
    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
        torch.cuda.set_device(dev_id)
    args.model.to(args.device)
    dm = DeviceMesh("cuda", torch.arange(0, args.world_size))

    print(f"device mesh {dm}")
    print("====================================")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompt = "Hey, are you conscious? Can you talk to me?"
    input = tokenizer(prompt, return_tensors="pt")
    
    input_ids = input["input_ids"]
    
    # logging the model modules, this is helpful for understanding/ eyeballing 
    # which layers are in the model/ candidates for TP.
    if rank==0:
        print_submodules(model)
    input_ids = input_ids.to(args.device)
    
    print("==============================================")
    TP_time_start = time.perf_counter()
    tp_llama(model,dm)
    TP_time_end = time.perf_counter()-TP_time_start
    print(f"TP time took {TP_time_end}s")
    print("==============================================")
    
    # model.to(args.device)

    
    dist.barrier()
    print(f"Rank {args.rank} completes")
    
    generate_time_start = time.perf_counter()
    outputs = model.generate(input_ids, max_new_tokens=30)
 
    generate_time_end = time.perf_counter()-generate_time_start
    print(f"Generate time took {generate_time_end}s")
    print("==============================================")
   
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)
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
    rank = int(os.environ["RANK"])
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
   
    backend = "nccl" if args.cuda else "gloo"
 
    run_all(args)
  
