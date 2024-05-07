# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_layoutLM.py

import argparse
import os

import torch
import torch.distributed as dist

from pippy import pipeline
from pippy import SplitPoint, annotate_split_points
from pippy.PipelineSchedule import ScheduleGPipe
from pippy import PipelineStage

from transformers import LayoutLMForMaskedLM, LayoutLMConfig

from hf_utils import generate_inputs_for_model, get_number_of_params


def add_split_points(layoutlm, nranks):
    # First stage carries the embedding layer
    annotate_split_points(
        layoutlm, {"layoutlm.embeddings": SplitPoint.END})
    # Last stage carries the LM head
    annotate_split_points(
        layoutlm, {"cls": SplitPoint.BEGINNING})
    # 12 Transformer layers divided over the rest 2 ranks
    layers_per_rank = layoutlm.config.num_hidden_layers // (nranks - 2)
    for i in range(1, nranks - 2):
        annotate_split_points(
            layoutlm, {f"layoutlm.encoder.layer.{i * layers_per_rank}": SplitPoint.BEGINNING})


def run(args):
    # Model configs
    config = LayoutLMConfig()
    print("Using device:", args.device)

    # Create model
    model_class = LayoutLMForMaskedLM
    model_name = "LayoutLMForMaskedLM"
    layoutlm = model_class(config)
    layoutlm.to(args.device)
    layoutlm.eval()
    if args.rank == 0:
        print(layoutlm.config)
        print(f"Total number of params = {get_number_of_params(layoutlm) // 10 ** 6}M")
        print(layoutlm)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, layoutlm, model_name, args.batch_size, args.device)
    input_ids = example_inputs["input_ids"]

    # Annotate split points
    add_split_points(layoutlm, args.world_size)

    # Create pipeline
    layoutlm_pipe = pipeline(
        layoutlm,
        num_chunks=args.chunks,
        example_args=(input_ids, ),
    )

    assert layoutlm_pipe.num_stages == args.world_size, f"nstages = {layoutlm_pipe.num_stages} nranks = {args.world_size}"
    if args.rank == 0:
        for i, sm in enumerate(layoutlm_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        layoutlm_pipe,
        args.rank,
        device=args.device,
    )

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, args.chunks)

    # Run
    if args.rank == 0:
        schedule.step(input_ids)
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
