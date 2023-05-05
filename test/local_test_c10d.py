# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.distributed as dist

from pippy.fx.passes import shape_prop
from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split


d_hid = 512
bs = 256

torch.manual_seed(0)


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = torch.mm(x, self.mm_param)
        # skip_connection = x
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param)
        x = self.lin(x)
        pipe_split()
        x = torch.relu(x)
        # x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin(x)
        x = torch.relu(x)
        return x


def run_worker(args):
    ec = ExampleCode()
    ec.to(args.device)
    ec_input = torch.randn(bs, d_hid, device=args.device)

    # Trace and cut
    ec_pipe = Pipe.from_tracing(ec, MultiUseParameterConfig.REPLICATE)
    gm = ec_pipe.split_gm
    if args.rank == 0:
        print(gm)
        gm.graph.print_tabular()

    # Use fake tensor for shape propagation
    # Since model itself may have been materialized, we need to use
    # `allow_non_fake_inputs`
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    # In reality, the fake input should be created from shape info (potentially
    # broadcast from Rank 0)
    fake_input = fake_mode.from_tensor(ec_input)
    sp = shape_prop.ShapeProp(gm)
    sp.propagate(fake_input)

    # Get input
    if args.rank == 0:
        x = ec_input
    else:

    # Compute
    y = submod(x)

    # Send to next stage
    dist.send(y, (args.rank + 1) % args.world_size)

    # Rank 0 checks result
    if args.rank == 0:
        # Get final output shape
        for node in gm.graph.nodes:
            if node.target == "output":
                break
        tensor_meta = node.meta["tensor_meta"]
        z = make_tensor_from_meta(tensor_meta, args.device)
        dist.recv(z, args.world_size - 1)
        ref_out = ec_pipe(ec_input)
        torch.testing.assert_close(z, ref_out)
        print(
            f"equivalence test passed {torch.sum(z)} ref {torch.sum(ref_out)}"
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
    args = parser.parse_args(args)

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = f"cuda:{dev_id}"
    else:
        args.device = "cpu"

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run_worker(args)


if __name__ == "__main__":
    main()


class LocalTestC10DTest(unittest.TestCase):
    def test_c10d(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
