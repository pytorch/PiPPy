# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_gpt2.py

import argparse
import os

import torch
import torch.distributed as dist

from pippy import pipeline
from pippy import SplitPoint, annotate_split_points
from pippy.PipelineSchedule import ScheduleGPipe
from pippy.PipelineStage import PipelineStage

from transformers import GPT2ForSequenceClassification, GPT2Config

from hf_utils import generate_inputs_for_model, get_number_of_params


def add_split_points(gpt2, nranks):
    decoders_per_rank = (gpt2.config.n_layer + nranks - 1) // nranks
    print(f"decoders_per_rank = {decoders_per_rank}")
    nstages = 1
    for i in range(1, gpt2.config.n_layer // decoders_per_rank):
        annotate_split_points(
            gpt2,
            {f'transformer.h.{i * decoders_per_rank}': SplitPoint.BEGINNING},
        )
        nstages += 1
    assert nstages == nranks, f"nstages = {nstages} nranks = {nranks}"


def run(args):
    # Model configs
    config = GPT2Config()
    config.n_embd = args.n_embd or config.n_embd
    config.n_layer = args.n_layer or config.n_layer
    config.n_head = args.n_head or config.n_head
    print("Using device:", args.device)

    # Create model
    model_class = GPT2ForSequenceClassification
    model_name = "GPT2ForSequenceClassification"
    gpt2 = model_class(config)
    gpt2.to(args.device)
    gpt2.eval()
    if args.rank == 0:
        print(gpt2.config)
        print(f"GPT-2 total number of params = {get_number_of_params(gpt2) // 10 ** 6}M")
        print(gpt2)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, gpt2, model_name, args.batch_size, args.device)

    if args.autosplit:
        # Automatic split
        from pippy import split_into_equal_size
        gpt2_pipe = pipeline(
            gpt2,
            num_chunks=args.chunks,
            example_args=(),
            example_kwargs=example_inputs,
            split_policy=split_into_equal_size(args.world_size),
        )
    else:
        # Manually annotate split points
        add_split_points(gpt2, args.world_size)
        gpt2_pipe = pipeline(
            gpt2,
            num_chunks=args.chunks,
            example_args=(),
            example_kwargs=example_inputs,
        )

    assert gpt2_pipe.num_stages == args.world_size
    if args.rank == 0:
        for i, sm in enumerate(gpt2_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        gpt2_pipe,
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
    # Note: this specific example requires: 1) a batch size that is divisible by
    # the number of chunks; 2) the division result (i.e. chunk size) must be 1,
    # otherwise padding token must be provided too (see GPT-2's forward function)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--autosplit', action="store_true")

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
