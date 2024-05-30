# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 gpt2_cpu_init.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe, SplitPoint

from transformers import GPT2ForSequenceClassification, GPT2Config


def run(args):
    # Model configs
    config = GPT2Config()

    # Create model on CPU
    model_class = GPT2ForSequenceClassification
    model_name = "GPT2ForSequenceClassification"
    gpt2 = model_class(config)
    gpt2.eval()
    if args.rank == 0:
        print(gpt2.config)
        print(gpt2)

    # Example input on CPU
    example_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, 512),  # bs x seq_len
        device="cpu",
        dtype=torch.int64,
        requires_grad=False,
    )

    # Split spec
    decoders_per_rank = (gpt2.config.n_layer + args.world_size - 1) // args.world_size
    print(f"decoders_per_rank = {decoders_per_rank}")
    split_spec = {
        f'transformer.h.{i * decoders_per_rank}': SplitPoint.BEGINNING
        for i in range(1, args.world_size)
    }

    # Create pipeline
    pipe = pipeline(
        gpt2,
        num_chunks=args.chunks,
        example_args=(example_input,),
        split_spec=split_spec,
    )

    assert pipe.num_stages == args.world_size, f"nstages = {pipe.num_stages} nranks = {args.world_size}"

    # Create schedule runtime
    stage = PipelineStage(
        pipe,
        args.rank,
        device=args.device,
    )

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, args.chunks)

    # Real input on GPU
    real_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, 512),  # bs x seq_len
        device=args.device,
        dtype=torch.int64,
        requires_grad=False,
    )

    # Run
    if args.rank == 0:
        schedule.step(real_input)
    else:
        out = schedule.step()

    print(f"Rank {args.rank} completes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=1)

    args = parser.parse_args()

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

    run(args)
