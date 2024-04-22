# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import pippy

import torch
import torch.distributed as dist
from pippy import pipe_split, pipeline, PipelineStage


pippy.microbatch._debug_mask_minibatches = True

d_hid = 512
batch_size = 256

torch.manual_seed(0)


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        # Test change of tensor creation device after tracing
        a = torch.ones(batch_size, d_hid, device=x.device)
        x = x + a
        x = torch.mm(x, self.mm_param)
        x = torch.relu(x)
        pipe_split()
        x = self.lin(x)
        x = torch.relu(x)
        return x


def run_worker(args):
    # Create module and trace model in CPU
    mod = ExampleCode()

    xe = torch.randn(batch_size, d_hid)

    pipe = pipeline(
        mod,
        args.chunks,
        example_args=(xe,),
    )

    # Create pipeline stages and move stage to GPU
    stage = PipelineStage(
        pipe,
        args.rank,
        device=args.device,
    )

    # Create real input on real device
    x = torch.randn(batch_size, d_hid, device=args.device)

    # Run
    if args.rank == 0:
        stage(x)
    elif args.rank == args.world_size - 1:
        out = stage()
    else:
        stage()

    dist.barrier()
    print(f"Rank {args.rank} completes")

    # Last rank checks result
    if args.rank == args.world_size - 1:
        mod.to(args.device)
        ref_out = mod(x)
        torch.testing.assert_close(out, ref_out)
        print(
            f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}"
        )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 2))
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


class TestFwd(unittest.TestCase):
    def test_fwd(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
