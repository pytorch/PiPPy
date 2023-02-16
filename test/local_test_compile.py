# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest
import pippy
from pippy import run_pippy
from pippy.IR import pipe_split

import torch

d_hid = 512
bs = 256

class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param)
        x = self.lin(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin(x)
        x = torch.relu(x)
        return {"out": x}


def run_master(_, args):
    ec = ExampleCode()
    ec.to(args.device)
    ec_input = torch.randn(bs, d_hid, device=args.device)

    # Create pipeline model
    pipe_ec = pippy.compile(
        ec,
        num_ranks=args.world_size,
        num_chunks=4,
        schedule=args.schedule,
        checkpoint=bool(args.checkpoint),
        _debug_mask_minibatches=True, # for numerical equivalence test only
    )

    # Warm up and correctness runs
    out = pipe_ec(ec_input)
    ref_out = ec(ec_input)

    # run with different chunk size to exercise microbatch and scheduling components
    torch.testing.assert_close(out["out"], ref_out["out"])
    print(
        f'equivalence test passed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}'
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
        "-s",
        "--schedule",
        type=str,
        default="FillDrain",
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    args = parser.parse_args(args)

    # Interleaved 1F1B uses less ranks than number of stages
    if args.schedule == "Interleaved1F1B":
        args.world_size = 2

    run_pippy(run_master, args)


if __name__ == "__main__":
    main()


class LocalTestCompileTest(unittest.TestCase):
    def test_compile(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--cuda",
            os.getenv("USE_CUDA", "0"),
            "--master_port",
            str(port),
        ]
        main(args)
