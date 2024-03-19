# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 2 pippy_t5.py

# Note: this example currently supports two ranks only due to:
# (1) the need of decoder_input_ids;
# (2) the `embed_tokens` module is shared between encoder and decoder. In the
# 2-rank case, we cut the model carefully so that `embed_tokens` is only used on
# rank 0.


import argparse
import os

import torch
import torch.distributed as dist

from pippy.IR import Pipe, SplitPoint, annotate_split_points
from pippy.PipelineStage import PipelineStage

from transformers import T5ForConditionalGeneration, T5Config

from hf_utils import generate_inputs_for_model, get_number_of_params


def add_split_points(t5, nranks):
    # Number of encoder layers: t5.config.num_layers
    # Number of decoder layers: t5.config.num_decoder_layers
    # 6 encoder layers, 6 decoder layers, 12 layers in total
    total_layers = t5.config.num_layers + t5.config.num_decoder_layers
    layers_per_rank = (total_layers + nranks - 1) // nranks
    print(f"Layers per rank = {layers_per_rank}")
    # Split encoder
    for i in range(1, t5.config.num_layers):
        if i % layers_per_rank == 0:
            annotate_split_points(
                t5, {f'encoder.block.{i}': SplitPoint.BEGINNING})
    # Split decoder
    for i in range(0, t5.config.num_decoder_layers):
        if i % layers_per_rank == 0:
            annotate_split_points(
                t5, {f'decoder.block.{i}': SplitPoint.BEGINNING})


def run(args):
    # Model configs
    config = T5Config()
    print("Using device:", args.device)

    # Create model
    model_class = T5ForConditionalGeneration
    model_name = "T5ForConditionalGeneration"
    t5 = model_class(config)
    t5.to(args.device)
    t5.eval()
    if args.rank == 0:
        print(t5.config)
        print(f"Total number of params = {get_number_of_params(t5) // 10 ** 6}M")
        print(t5)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, t5, model_name, args.batch_size, args.device)

    # Annotate split points
    add_split_points(t5, args.world_size)

    # Create pipeline
    t5_pipe = Pipe.from_tracing(
        t5,
        num_chunks=args.chunks,
        example_args=(),
        example_kwargs=example_inputs,
    )
    assert t5_pipe.num_stages == args.world_size, f"pipe stages: {t5_pipe.num_stages}"
    if args.rank == 0:
        for i, sm in enumerate(t5_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        t5_pipe,
        args.rank,
        device=args.device,
    )

    # Run
    if args.rank == 0:
        stage(**example_inputs)
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
