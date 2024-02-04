# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

from transformers import BertLMHeadModel, BertConfig

import torch
import torch.distributed as dist

import pippy
from pippy.hf import PiPPyHFTracer
from pippy.microbatch import sum_reducer, TensorChunkSpec
from pippy.PipelineDriver import PipelineDriverFillDrain


def run_worker(args):
    torch.manual_seed(25)

    bert = BertLMHeadModel(BertConfig(is_decoder=True))
    bert.to(args.device)
    print(bert.config)

    output_chunk_spec = (TensorChunkSpec(0), sum_reducer)

    split_policy = pippy.split_into_equal_size(args.world_size)

    # print("Instantiating BERT pipeline...")
    chunks = args.chunks or args.world_size
    seq_length = 16
    bs = 1 * chunks
    sample_input = torch.zeros(bs, seq_length, dtype=torch.long, device=args.device).random_(bert.config.vocab_size)
    stage = pippy.compile_stage(
        bert,
        args.rank,
        args.world_size,
        num_chunks=args.chunks,
        device=args.device,
        group=None,
        example_inputs=[sample_input, ],
        split_policy=split_policy,
        tracer=PiPPyHFTracer()
    )



if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument("--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost"))
    parser.add_argument("--master_port", type=str, default=os.getenv("MASTER_PORT", "29500"))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()))

    args = parser.parse_args()

    # args.pp_group_size = 4
    # assert args.world_size % args.pp_group_size == 0
    # args.dp_group_size = args.world_size // args.pp_group_size

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run_worker(args)
