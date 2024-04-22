# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch
import torch.distributed as dist

from pippy.IR import annotate_split_points, pipeline, SplitPoint
from pippy.PipelineSchedule import ScheduleInterleaved1F1B, ScheduleLoopedBFS
from pippy.PipelineStage import PipelineStage

# Using same key words as single-stage tests for convenience in CI.
schedule_map = {
    "gpipe": ScheduleLoopedBFS,  # BFS is a generalization of gpipe
    "1f1b": ScheduleInterleaved1F1B,
}

d_hid = 16
n_layers = 8
batch_size = 16

torch.manual_seed(0)


class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super(MLPModule, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class TransformerLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[MLPModule(d_hid) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def run_worker(args):
    model = TransformerLike().to(args.device)
    x = torch.randn(batch_size, d_hid, device=args.device)
    target = torch.randn(batch_size, d_hid, device=args.device)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Two stages per rank
    num_stages = 2 * args.world_size

    # Split model into stages
    layers_per_stage = n_layers // num_stages
    for stage_idx in range(1, num_stages):
        annotate_split_points(
            model,
            {f"layers.{layers_per_stage * stage_idx}": SplitPoint.BEGINNING},
        )

    pipe = pipeline(
        model,
        args.chunks,
        (x,),
    )
    assert pipe.num_stages == num_stages, f"{pipe.num_stages} != {num_stages}"

    # Collect my stages
    stages = []
    for stage_idx in range(pipe.num_stages):
        if stage_idx % args.world_size == args.rank:
            stage = PipelineStage(pipe, stage_idx, device=args.device)
            stages.append(stage)

    # Attach to an interleaving schedule
    ScheduleClass = schedule_map[args.schedule]
    schedule = ScheduleClass(stages, args.chunks, loss_fn=loss_fn)

    # Run
    if args.rank == 0:
        schedule.step(x)
    elif args.rank == args.world_size - 1:
        losses = []
        out = schedule.step(target=target, losses=losses)
    else:
        schedule.step()

    dist.barrier()
    print(f"Rank {args.rank} completes")

    # Last rank checks result
    if args.rank == args.world_size - 1:
        ref_out = model(x)
        ref_loss = loss_fn(ref_out, target)
        pipe_loss = sum(losses)
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(pipe_loss, ref_loss)
        print(
            f"equivalence test passed pipe_loss={pipe_loss} ref_loss={ref_loss}"
        )


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
        default="1f1b",
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


class TestInterleave(unittest.TestCase):
    def test_interleave(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
