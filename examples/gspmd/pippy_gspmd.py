# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch
import torch.autograd.profiler_legacy

import pippy
import pippy.fx
from pippy import run_pippy
from pippy.IR import pipe_split


pippy.fx.Tracer.proxy_buffer_attributes = True

d_hid = 512
bs = 500

class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)
        self.register_buffer("buffer", torch.randn(bs + 100, d_hid))

    def forward(self, x):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param) + self.buffer[: x.shape[0]]
        x = self.lin(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin(x)
        x = torch.relu(x)
        return {"out": x}


def run_gspmd(pp_ranks, args):
    # Make sure all ranks have the same seed
    torch.manual_seed(5)
    ec = ExampleCode()
    ec_input = torch.randn(bs, d_hid, device=args.device)
    chunks = 5

    pipe_driver, stage_mod = pippy.all_compile(
        ec,
        args.world_size,
        chunks,
        schedule=args.schedule,
        _debug_mask_minibatches=True,   # For numeric check only
    )
    print(
        f"Rank {args.rank}: {stage_mod}"
    )

    # PiPPy run
    if pipe_driver:
        out = pipe_driver(ec_input)

    # Reference run
    ec.to(args.device)
    ref_out = ec(ec_input)

    # Numeric check
    if pipe_driver:
        torch.testing.assert_close(out["out"], ref_out["out"])
        print(
            f'equivalence test passed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}'
        )

    print(f"Rank {args.rank} completed")


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
        "-s",
        "--schedule",
        type=str,
        default="FillDrain",
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    args = parser.parse_args(args)

    # Interleaved 1F1B uses less ranks than number of stages
    if args.schedule == "Interleaved1F1B":
        args.world_size = 2

    args.gspmd = 1
    run_pippy(run_gspmd, args)


if __name__ == "__main__":
    main()


class LocalTestGspmdTest(unittest.TestCase):
    def test_gspmd(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
