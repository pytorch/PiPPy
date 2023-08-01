# Copyright (c) Meta Platforms, Inc. and affiliates
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pippy
import argparse
import os
import torch.distributed as dist
from pippy.microbatch import sum_reducer, TensorChunkSpec

from transformers import OPTConfig, OPTModel

def get_model_size(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# Get process group for ranks in a pipeline
def get_pp_subgroup(args):
    my_pp_rank = args.rank // args.dp_group_size
    my_dp_rank = args.rank % args.dp_group_size
    for dp_rank in range(0, args.dp_group_size):
        pp_group_ranks = list(
            range(dp_rank, args.world_size, args.dp_group_size)
        )
        pp_group = dist.new_group(ranks=pp_group_ranks)
        if dp_rank == my_dp_rank:
            my_pp_group = pp_group
    print(f"Rank {args.rank} done getting pp group")
    return my_pp_group, my_pp_rank


# Get DP process group for ranks with the same stage
def get_dp_subgroup(args):
    my_pp_rank = args.rank // args.dp_group_size
    my_dp_rank = args.rank % args.dp_group_size
    for pp_rank in range(0, args.pp_group_size):
        dp_group_ranks = list(
            range(
                pp_rank * args.dp_group_size, (pp_rank + 1) * args.dp_group_size
            )
        )
        dp_group = dist.new_group(ranks=dp_group_ranks)
        if pp_rank == my_pp_rank:
            my_dp_group = dp_group
    print(f"Rank {args.rank} done getting dp group")
    return my_dp_group, my_dp_rank



def run_worker(args):
    torch.manual_seed(42)

    # Get DP and PP sub process groups
    dp_group, dp_rank = get_dp_subgroup(args)
    pp_group, pp_rank = get_pp_subgroup(args)

    configuration = OPTConfig()
    opt = OPTModel(configuration)
    # opt = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16).cuda()
    device = torch.device("cuda")
    opt.to(device)
    print(f"OPT total number of params = {get_model_size(opt) // 10 ** 6}M")
    print(type(opt), get_model_size(opt))  # size: 125,239,296 ~ 125M params

    output_chunk_spec = (TensorChunkSpec(0), sum_reducer)
    bs = 1 * args. chunks
    seq_length = 16
    opt_input_dict = {
        'input_ids': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(opt.config.vocab_size),
        'labels': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(opt.config.vocab_size),
        'attention_mask': torch.ones(bs, seq_length, device=device)
        }
    concrete_args = pippy.create_default_args(
        opt,
        except_keys=opt_input_dict.keys(),
    )
    sample_input = torch.zeros(bs, seq_length, dtype=torch.long, device=args.device).random_(opt.config.vocab_size)

    stage = pippy.compile_stage(
        opt,
        pp_rank,
        args.pp_group_size,
        args.chunks,
        args.device,
        pp_group,
        [],
        output_chunk_spec=output_chunk_spec,
        concrete_args=concrete_args,
    )

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)

    # prompt = "i am a lost boy"
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    # generated_ids = model.generate(input_ids)
    # print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 3))
    )
    parser.add_argument(
        "--rank", type=int, default=int(os.getenv("RANK", -1))
    )
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10
    )
    parser.add_argument(
        "--batch_size", type=int, default=10
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    parser.add_argument("--checkpoint_epochs", type=int, default=5)
    parser.add_argument(
        "--chunks", type=int, default=4,
    )
    args = parser.parse_args()

    args.pp_group_size = 4
    assert args.world_size % args.pp_group_size == 0
    args.dp_group_size = args.world_size // args.pp_group_size

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run_worker(args)
