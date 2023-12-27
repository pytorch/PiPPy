# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 4 bert_cpu_init.py

import argparse
import os

import torch
import torch.distributed as dist

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage

from transformers import BertForMaskedLM, BertConfig


def add_split_points(bert, nranks):
    layers_per_rank = bert.config.num_hidden_layers // nranks
    for i in range(1, nranks):
        annotate_split_points(
            bert, {f"bert.encoder.layer.{i * layers_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING})


def run(args):
    # Model configs
    config = BertConfig()

    # Create model on CPU
    model_class = BertForMaskedLM
    model_name = "BertForMaskedLM"
    bert = model_class(config)
    bert.eval()
    if args.rank == 0:
        print(bert.config)
        print(bert)

    # Example input on CPU
    example_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, 512),  # bs x seq_len
        device="cpu",
        dtype=torch.int64,
        requires_grad=False,
    )

    # Annotate split points
    add_split_points(bert, args.world_size)

    # Create pipeline
    bert_pipe = Pipe.from_tracing(
        bert,
        num_chunks=args.chunks,
        example_args=(example_input,),
    )
    nstages = len(list(bert_pipe.split_gm.children()))
    assert nstages == args.world_size, f"nstages = {nstages} nranks = {args.world_size}"

    # Create schedule runtime
    stage = PipelineStage(
        bert_pipe,
        args.rank,
        device=args.device,
    )
    stage._move_ops_to_device(args.device)

    # Real input on GPU
    real_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, 512),  # bs x seq_len
        device=args.device,
        dtype=torch.int64,
        requires_grad=False,
    )

    # Run
    if args.rank == 0:
        stage(real_input)
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
