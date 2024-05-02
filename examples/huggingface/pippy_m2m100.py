# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 3 pippy_m2m100.py

import argparse
import os

import torch
import torch.distributed as dist

from pippy import pipeline
from pippy import SplitPoint, annotate_split_points
from pippy.PipelineSchedule import ScheduleGPipe
from pippy.PipelineStage import PipelineStage

from transformers import M2M100ForConditionalGeneration, M2M100Config

from hf_utils import generate_inputs_for_model, get_number_of_params


def add_split_points(m2m100, nranks):
    # First rank takes encoder and shared embedding
    # Second rank takes decoder layers
    annotate_split_points(
        m2m100, {"model.decoder.layers.0": SplitPoint.BEGINNING})
    # Last rank takes LM head
    annotate_split_points(
        m2m100, {"lm_head": SplitPoint.BEGINNING})


def run(args):
    # Model configs
    config = M2M100Config()
    print("Using device:", args.device)

    # Create model
    model_class = M2M100ForConditionalGeneration
    model_name = "M2M100ForConditionalGeneration"
    m2m100 = model_class(config)
    m2m100.to(args.device)
    m2m100.eval()
    if args.rank == 0:
        print(m2m100.config)
        print(f"Total number of params = {get_number_of_params(m2m100) // 10 ** 6}M")
        print(m2m100)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, m2m100, model_name, args.batch_size, args.device)

    # Annotate split points
    add_split_points(m2m100, args.world_size)

    # Create pipeline
    m2m100_pipe = pipeline(
        m2m100,
        num_chunks=args.chunks,
        example_args=(),
        example_kwargs=example_inputs,
    )

    assert m2m100_pipe.num_stages == args.world_size, f"nstages = {m2m100_pipe.num_stages} nranks = {args.world_size}"
    if args.rank == 0:
        for i, sm in enumerate(m2m100_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        m2m100_pipe,
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

    print(f"Rank {args.rank} completes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 3)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
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
