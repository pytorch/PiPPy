# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_distilBert.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe, SplitPoint

from transformers import DistilBertForMaskedLM, DistilBertConfig

from hf_utils import generate_inputs_for_model, get_number_of_params


def run(args):
    # Model configs
    config = DistilBertConfig()
    print("Using device:", args.device)

    # Create model
    model_class = DistilBertForMaskedLM
    model_name = "DistilBertForMaskedLM"
    distilbert = model_class(config)
    distilbert.to(args.device)
    distilbert.eval()
    if args.rank == 0:
        print(distilbert.config)
        print(f"Total number of params = {get_number_of_params(distilbert) // 10 ** 6}M")
        print(distilbert)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, distilbert, model_name, args.batch_size, args.device)
    input_ids = example_inputs["input_ids"]

    # Split points
    split_spec = {}
    # The first rank carries the embedding layer
    split_spec[f"distilbert.embeddings"] = SplitPoint.END
    # 6 Transformer layers divided over the rest 3 ranks
    layers_per_rank = distilbert.config.num_hidden_layers // (args.world_size - 1)
    for i in range(1, args.world_size - 1):
        split_spec[f"distilbert.transformer.layer.{i * layers_per_rank}"] = SplitPoint.BEGINNING

    # Create pipeline
    pipe = pipeline(
        distilbert,
        num_chunks=args.chunks,
        example_args=(input_ids, ),
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
    parser.add_argument('--batch_size', type=int, default=256)
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
