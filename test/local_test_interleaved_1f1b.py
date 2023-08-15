# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch
import torch.distributed as dist
import pippy

from pippy.compile import compile_stage
from pippy.IR import pipe_split


d_hid = 512
chunk_size = 256

torch.manual_seed(0)


class ExampleCode1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, target):
        x = torch.mm(x, self.mm_param)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param)
        x = self.lin(x)
        pipe_split()
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin(x)
        x = torch.relu(x)
        loss = self.mse_loss(x, target)
        return {"logits": x, "loss": loss}

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


class ExampleCode(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)
        self.mlp4 = MLPModule(d_hid)
        self.mlp5 = MLPModule(d_hid)
        self.mlp6 = MLPModule(d_hid)
        self.mlp7 = MLPModule(d_hid)
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, target):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        pipe_split()
        x = self.mlp4(x)
        pipe_split()
        x = self.mlp5(x)
        pipe_split()
        x = self.mlp6(x)
        pipe_split()
        x = self.mlp7(x)
        loss = self.mse_loss(x, target)
        return {"logits": x, "loss": loss}


def run_worker(args):
    ec = ExampleCode(d_hid)
    ec.to(args.device)
    ec.train()

    ec_x = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)
    target = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)

    tot_nstages = args.world_size * 2
    my_stage_ids = list(range(args.rank, tot_nstages, args.world_size))
    print(f"Rank {args.rank} has stages {my_stage_ids}")

    my_stages = compile_stage(
        ec,
        my_stage_ids,
        tot_nstages,
        args.chunks,
        args.device,
        None,
        [ec_x, target],
        schedule="1F1B",
    )

    interleaver = pippy.StageInterleaver(my_stages)

    # Run
    if interleaver.has_first():
        interleaver(ec_x)
    elif interleaver.has_last():
        out = interleaver(target)
    else:
        interleaver()

    #dist.barrier()
    print(f"Rank {args.rank} completes")

    """
    # Last rank checks result
    if interleaver.has_last():
        ref_out = ec(ec_x, target)
        torch.testing.assert_close(out, ref_out)
        print(
            f"equivalence test passed, loss = {out['loss']}, ref loss = {ref_out['loss']}"
        )
    """


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
        default=8,
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


class LocalTestC10DBwdTest(unittest.TestCase):
    def test_c10d_bwd(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
