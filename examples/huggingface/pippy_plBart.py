# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_plBart.py

import argparse
import os

import torch
import torch.distributed as dist

from pippy.IR import Pipe, SplitPoint, annotate_split_points
from pippy.PipelineStage import PipelineStage

from transformers import PLBartForCausalLM, PLBartConfig

from hf_utils import generate_inputs_for_model, get_number_of_params


def add_split_points(plbart, nranks):
    # The first rank carries embedding
    annotate_split_points(
        plbart, {f"model.decoder.embed_positions": SplitPoint.END})
    # The last rank carries LM head
    annotate_split_points(
        plbart, {"lm_head": SplitPoint.BEGINNING})
    # The rest ranks divide 6 layers
    layers_per_rank = plbart.config.num_hidden_layers // (nranks - 2)
    for i in range(1, nranks - 2):
        annotate_split_points(
            plbart, {f"model.decoder.layers.{i * layers_per_rank}": SplitPoint.BEGINNING})


def run(args):
    # Model configs
    config = PLBartConfig()
    print("Using device:", args.device)

    # Create model
    model_class = PLBartForCausalLM
    model_name = "PLBartForCausalLM"
    plbart = model_class(config)
    plbart.to(args.device)
    plbart.eval()
    if args.rank == 0:
        print(plbart.config)
        print(f"Total number of params = {get_number_of_params(plbart) // 10 ** 6}M")
        print(plbart)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, plbart, model_name, args.batch_size, args.device)
    input_ids = example_inputs["input_ids"]

    # Annotate split points
    add_split_points(plbart, args.world_size)

    # Create pipeline
    plbart_pipe = Pipe.from_tracing(
        plbart,
        num_chunks=args.chunks,
        example_args=(input_ids, ),
    )
    nstages = len(list(plbart_pipe.split_gm.children()))
    assert nstages == args.world_size, f"nstages = {nstages} nranks = {args.world_size}"
    if args.rank == 0:
        for i, sm in enumerate(plbart_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        plbart_pipe,
        args.rank,
        device=args.device,
    )

    # Run
    if args.rank == 0:
        stage(input_ids)
    elif args.rank == args.world_size - 1:
        out = stage()
    else:
        stage()

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
    parser.add_argument('--batch_size', type=int, default=16)
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
