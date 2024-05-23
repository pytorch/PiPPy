# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 pippy_convBert.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe, SplitPoint

from transformers import ConvBertForMaskedLM, ConvBertConfig

from hf_utils import generate_inputs_for_model, get_number_of_params


def run(args):
    # Model configs
    config = ConvBertConfig()
    print("Using device:", args.device)

    # Create model
    model_class = ConvBertForMaskedLM
    model_name = "ConvBertForMaskedLM"
    convbert = model_class(config)
    convbert.to(args.device)
    convbert.eval()
    if args.rank == 0:
        print(convbert.config)
        print(f"Total number of params = {get_number_of_params(convbert) // 10 ** 6}M")
        print(convbert)

    # Input configs
    example_inputs = generate_inputs_for_model(
        model_class, convbert, model_name, args.batch_size, args.device)
    input_ids = example_inputs["input_ids"]

    # Split points
    # The first rank takes embedding
    split_spec = {}
    split_spec["convbert.embeddings"] = SplitPoint.END
    # The last rank takes generation
    split_spec["generator_predictions"] = SplitPoint.BEGINNING
    # The rest ranks divide encoder layers
    layers_per_rank = convbert.config.num_hidden_layers // (args.world_size - 2)
    for i in range(1, args.world_size - 2):
            split_spec[f"convbert.encoder.layer.{i * layers_per_rank}"] = SplitPoint.BEGINNING

    # Create pipeline
    pipe = pipeline(
        convbert,
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
