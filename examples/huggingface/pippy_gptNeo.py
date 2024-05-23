# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_gptNeo.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe, SplitPoint

from transformers import GPTNeoForCausalLM, GPTNeoConfig

from hf_utils import generate_inputs_for_model, get_number_of_params


def run(args):
    # Model configs
    config = GPTNeoConfig()
    print("Using device:", args.device)

    # Create model
    model_class = GPTNeoForCausalLM
    model_name = "GPTNeoForCausalLM"
    gptneo = model_class(config)
    gptneo.to(args.device)
    gptneo.eval()
    if args.rank == 0:
        print(gptneo.config)
        print(f"Total number of params = {get_number_of_params(gptneo) // 10 ** 6}M")
        print(gptneo)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, gptneo, model_name, args.batch_size, args.device)

    # Annotate split points
    layers_per_rank = (gptneo.config.num_layers + args.world_size - 1) // args.world_size
    print(f"decoders_per_rank = {layers_per_rank}")
    split_spec = {
        f'transformer.h.{i * layers_per_rank}': SplitPoint.BEGINNING
        for i in range(1, args.world_size)
    }

    # Create pipeline
    pipe = pipeline(
        gptneo,
        num_chunks=args.chunks,
        example_args=(),
        example_kwargs=example_inputs,
        split_spec=split_spec,
    )

    assert pipe.num_stages == args.world_size, f"nstages = {pipe.num_stages} nranks = {args.world_size}"
    smod = pipe.get_stage_module(args.rank)
    print(f"Pipeline stage {args.rank} {get_number_of_params(smod) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        pipe,
        args.rank,
        device=args.device,
    )

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, args.chunks)

    # Run
    if args.rank == 0:
        schedule.step(**example_inputs)
    else:
        out = schedule.step()

    dist.destroy_process_group()
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
    parser.add_argument('--batch_size', type=int, default=32)
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
