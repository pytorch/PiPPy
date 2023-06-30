# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import pippy

import torch
import torch.distributed as dist
import torch.optim as optim
from pippy.IR import pipe_split, TrivialLossWrapper
from torch.nn.parallel import DistributedDataParallel


d_hid = 512
chunk_size = 256

torch.manual_seed(0)


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = torch.mm(x, self.mm_param0)
        skip_connection = x
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin1(x)
        x = torch.relu(x)
        return x


def get_dp_subgroup(args):
    my_pp_rank = args.rank // args.dp_group_size
    my_dp_rank = args.rank % args.dp_group_size
    for pp_rank in range(0, args.pp_group_size):
        dp_group_ranks = list(
            range(
                pp_rank * args.dp_group_size, (pp_rank + 1) * args.dp_group_size
            )
        )
        dp_group = dist.new_group(ranks=dp_group_ranks)
        if pp_rank == my_pp_rank:
            my_dp_group = dp_group
    print(f"Rank {args.rank} done getting dp group")
    return my_dp_group, my_dp_rank


def get_pp_subgroup(args):
    my_pp_rank = args.rank // args.dp_group_size
    my_dp_rank = args.rank % args.dp_group_size
    for dp_rank in range(0, args.dp_group_size):
        pp_group_ranks = list(
            range(dp_rank, args.world_size, args.dp_group_size)
        )
        pp_group = dist.new_group(ranks=pp_group_ranks)
        if dp_rank == my_dp_rank:
            my_pp_group = pp_group
    print(f"Rank {args.rank} done getting pp group")
    return my_pp_group, my_pp_rank


def run_worker(args):
    ec = ExampleCode()
    loss_fn = torch.nn.MSELoss(reduction="sum")
    ec_with_loss = TrivialLossWrapper(ec, loss_fn)
    ec_with_loss.to(args.device)

    ec_x = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)
    target = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)

    # Get DP and PP sub process groups
    dp_group, dp_rank = get_dp_subgroup(args)
    pp_group, pp_rank = get_pp_subgroup(args)

    stage = pippy.compile_stage(
        ec_with_loss,
        pp_rank,
        args.pp_group_size,
        args.chunks,
        args.device,
        pp_group,
        [ec_x, target],
    )

    # Wrap stage module with DDP
    stage.submod = DistributedDataParallel(
        stage.submod,
        process_group=dp_group,
    )

    # Create an optimizer for stage submodule's parameters
    optimizer = optim.SGD(stage.submod.parameters(), lr=1e-3, momentum=0.9)

    for _ in range(2):
        # Zero gradients
        optimizer.zero_grad()

        # Run
        if pp_rank == 0:
            stage(ec_x)
        elif pp_rank == args.pp_group_size - 1:
            stage(target)
        else:
            stage()

        # Take an optimization step
        optimizer.step()

    dist.barrier()
    print(f"Rank {args.rank} completes")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 8))
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

    args.pp_group_size = 4
    assert args.world_size % args.pp_group_size == 0
    args.dp_group_size = args.world_size // args.pp_group_size
    if args.rank == 0:
        print(
            f"PP group size = {args.pp_group_size}, DP group size = {args.dp_group_size}"
        )

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


class LocalTestC10dDDPTest(unittest.TestCase):
    def test_c10d_ddp(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
