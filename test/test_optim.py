# Copyright (c) Meta Platforms, Inc. and affiliates
# Run this test with:
# torchrun --nproc-per-node 4 test/local_test_optim.py

import argparse
import os
import unittest

import torch
import torch.distributed as dist
import torch.optim as optim

from pippy.IR import pipe_split, pipeline
from pippy.PipelineSchedule import Schedule1F1B, ScheduleGPipe
from pippy.PipelineStage import PipelineStage


schedule_map = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B,
}

d_hid = 512
batch_size = 256

torch.manual_seed(0)


# Basic example
class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = torch.mm(x, self.mm_param0)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        pipe_split()
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin2(x)
        x = torch.relu(x)
        return x


def run_worker(args):
    mod = ExampleCode()
    mod.to(args.device)

    x = torch.randn(batch_size, d_hid, device=args.device)
    target = torch.randn(batch_size, d_hid, device=args.device)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    pipe = pipeline(
        mod,
        args.chunks,
        example_args=(x,),
    )

    stage = PipelineStage(
        pipe,
        args.rank,
        device=args.device,
    )

    # Attach to a schedule
    ScheduleClass = schedule_map[args.schedule]
    schedule = ScheduleClass(stage, args.chunks, loss_fn=loss_fn)

    # Create an optimizer for stage submodule's parameters
    optimizer = optim.SGD(stage.submod.parameters(), lr=1e-3, momentum=0.9)

    for _ in range(2):
        # Zero gradients
        optimizer.zero_grad()

        # Run
        if args.rank == 0:
            schedule.step(x)
        elif args.rank == args.world_size - 1:
            losses = []
            out = schedule.step(target=target, losses=losses)
        else:
            schedule.step()

        # Take an optimization step
        optimizer.step()

    dist.barrier()
    print(f"Rank {args.rank} completes")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="gpipe",
        choices=schedule_map.keys(),
    )
    args = parser.parse_args(args)

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

    run_worker(args)


if __name__ == "__main__":
    main()


class TestOptimTest(unittest.TestCase):
    def test_optim(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
