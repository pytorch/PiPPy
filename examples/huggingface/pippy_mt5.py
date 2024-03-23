# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_mt5.py

import argparse
import os

import torch
import torch.distributed as dist

from pippy import pipeline
from pippy.IR import SplitPoint, annotate_split_points
from pippy.PipelineStage import PipelineStage

from transformers import MT5ForConditionalGeneration, MT5Config

from hf_utils import generate_inputs_for_model, get_number_of_params


def add_split_points(mt5, nranks):
    # Number of encoder layers (mt5.config.num_layers): 8
    # Number of decoder layers (mt5.config.num_decoder_layers): 8
    # 16 layers in total
    total_layers = mt5.config.num_layers + mt5.config.num_decoder_layers
    layers_per_rank = (total_layers + nranks - 1) // nranks
    print(f"Layers per rank = {layers_per_rank}")
    nstages = 1
    # Split encoder
    for i in range(1, mt5.config.num_layers // layers_per_rank):
        annotate_split_points(
            mt5, {f'encoder.block.{i * layers_per_rank}': SplitPoint.BEGINNING})
        nstages += 1
    # Split at the boundary of encoder and decoder
    annotate_split_points(
        mt5, {f'decoder.embed_tokens': SplitPoint.BEGINNING})
    nstages += 1
    # Split decoder
    for i in range(1, mt5.config.num_decoder_layers // layers_per_rank):
        annotate_split_points(
            mt5, {f'decoder.block.{i * layers_per_rank}': SplitPoint.BEGINNING})
        nstages += 1
    assert nstages == nranks, f"nstages = {nstages} nranks = {nranks}"


def run(args):
    # Model configs
    config = MT5Config()
    print("Using device:", args.device)

    # Create model
    model_class = MT5ForConditionalGeneration
    model_name = "MT5ForConditionalGeneration"
    mt5 = model_class(config)
    mt5.to(args.device)
    mt5.eval()
    if args.rank == 0:
        print(mt5.config)
        print(f"Total number of params = {get_number_of_params(mt5) // 10 ** 6}M")
        print(mt5)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, mt5, model_name, args.batch_size, args.device)

    # Annotate split points
    add_split_points(mt5, args.world_size)

    # Create pipeline
    mt5_pipe = pipeline(
        mt5,
        num_chunks=args.chunks,
        example_args=(),
        example_kwargs=example_inputs,
    )
    nstages = len(list(mt5_pipe.split_gm.children()))
    assert nstages == args.world_size, f"nstages = {nstages} nranks = {args.world_size}"
    if args.rank == 0:
        for i, sm in enumerate(mt5_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        mt5_pipe,
        args.rank,
        device=args.device,
    )

    # Run
    if args.rank == 0:
        stage(example_inputs["input_ids"])
    elif args.rank == 1:
        stage(example_inputs["decoder_input_ids"])
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
