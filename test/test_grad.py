# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import copy
import os
import unittest

import torch
import torch.distributed as dist

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


# MLP example
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


class MultiMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)

    def forward(self, x):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        return x


def run_worker(args):
    mod = MultiMLP()
    mod.to(args.device)

    ref_mod = copy.deepcopy(mod)
    x = torch.randn(batch_size, d_hid, device=args.device)
    with torch.no_grad():
        y = ref_mod(x)
        # Add a small perturbation
        target = y + torch.randn(batch_size, d_hid, device=args.device)

    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Run reference
    ref_out = ref_mod(x)
    ref_loss = loss_fn(ref_out, target)
    ref_loss.backward()

    # Create a pipeline
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
    print(f"Using {ScheduleClass.__name__}")
    schedule = ScheduleClass(stage, args.chunks, loss_fn=loss_fn)

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
        # Check output
        torch.testing.assert_close(out, ref_out)
        print("Output test passed")
        # Check loss
        # Since the reduction used in the loss function above is "sum", we use
        # "sum" here to reduce microbatch losses into a single value too.
        pipe_loss = sum(losses)
        torch.testing.assert_close(pipe_loss, ref_loss)
        print("Loss test passed")

    # Every rank checks gradients
    stage_module = pipe.get_stage_module(args.rank)
    for name, p in stage_module.named_parameters():
        ref_p = ref_mod.get_parameter(name)
        try:
            torch.testing.assert_close(p.grad, ref_p.grad)
        except AssertionError:
            print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
            raise
    print(f"Rank {args.rank} Gradient test passed")


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


class TestGrad(unittest.TestCase):
    def test_grad(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
