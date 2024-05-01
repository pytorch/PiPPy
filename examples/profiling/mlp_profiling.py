# Copyright (c) Meta Platforms, Inc. and affiliates
# Run command:
# torchrun --nproc-per-node 4 mlp_profiling.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

from pippy.compile import compile_stage
from pippy import pipe_split


d_hid = 1024
chunk_size = 1024

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


class ExampleCode(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, target):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        loss = self.mse_loss(x, target)
        return {"logits": x, "loss": loss}


def run_worker(args):
    ec = ExampleCode(d_hid)
    ec.to(args.device)
    ec.train()

    ec_x = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)
    target = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)

    stage = compile_stage(
        ec,
        args.rank,
        args.world_size,
        args.chunks,
        args.device,
        None,
        [ec_x, target],
    )

    # Run
    for _ in range(10):
        if args.rank == 0:
            out = stage(ec_x)
        elif args.rank == args.world_size - 1:
            out = stage(target)
        else:
            stage()

    dist.barrier()
    print(f"Rank {args.rank} warmup completes")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(20):
            if args.rank == 0:
                out = stage(ec_x)
            elif args.rank == args.world_size - 1:
                out = stage(target)
            else:
                stage()

    print(f"Rank {args.rank} profiling run completed")
    prof.export_chrome_trace(
        f"{os.path.splitext(os.path.basename(__file__))[0]}_{args.rank}.json"
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
