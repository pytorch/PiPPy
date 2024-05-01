# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 2 pippy_unet.py

import argparse
import os

import torch
import torch.distributed as dist

from pippy import pipeline
from pippy import SplitPoint, annotate_split_points
from pippy.PipelineSchedule import ScheduleGPipe
from pippy.PipelineStage import PipelineStage

from diffusers import UNet2DModel

from hf_utils import get_number_of_params


def run(args):
    print("Using device:", args.device)

    # Create model
    # See https://github.com/huggingface/diffusers?tab=readme-ov-file#quickstart
    unet = UNet2DModel.from_pretrained("google/ddpm-cat-256")
    unet.to(args.device)
    unet.eval()
    if args.rank == 0:
        print(f"Total number of params = {get_number_of_params(unet) // 10 ** 6}M")
        print(unet)

    # Input configs
    sample_size = unet.config.sample_size
    noise = torch.randn((args.batch_size, 3, sample_size, sample_size), device=args.device)
    timestep = 1

    # Split model into two stages:
    #   Stage 0: down_blocks + mid_block
    #   Stage 2: up_blocks
    annotate_split_points(unet, {"mid_block": SplitPoint.END})

    # Create pipeline
    unet_pipe = pipeline(
        unet,
        num_chunks=args.chunks,
        example_args=(noise, timestep),
    )

    assert unet_pipe.num_stages == args.world_size, f"nstages = {unet_pipe.num_stages} nranks = {args.world_size}"
    if args.rank == 0:
        for i, sm in enumerate(unet_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        unet_pipe,
        args.rank,
        device=args.device,
    )

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, args.chunks)

    # Run
    if args.rank == 0:
        schedule.step(noise)
    else:
        out = schedule.step()

    print(f"Rank {args.rank} completes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 2)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
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
