# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import pippy

import torch
import torch.distributed as dist
from pippy.IR import pipe_split
from torch.nn.parallel import DistributedDataParallel


d_hid = 512
chunk_size = 256


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.loss_fn = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, target):
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
        loss = self.loss_fn(x, target)
        return {"loss": loss}


def create_model() -> torch.nn.Module:
    # Fix a seed such that models are created the same
    torch.manual_seed(42)
    ec = ExampleCode()
    return ec


# Get process group for ranks in a pipeline
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


# Get DP process group for ranks with the same stage
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


# Main program
def run_worker(args):
    ec_with_loss = create_model()
    ec_with_loss.to(args.device)

    input = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)
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
        [input, target],
    )

    # Form a map from original qualname to param for equivalence check later
    pipe_params = {}
    for qualname, param in stage.submod.named_parameters():
        origin_name = stage.submod.remap_qualname(qualname)
        pipe_params[origin_name] = param

    # Wrap stage module with DDP
    stage.submod = DistributedDataParallel(
        stage.submod,
        process_group=dp_group,
    )

    # Run
    if pp_rank == 0:
        stage(input)
    elif pp_rank == args.pp_group_size - 1:
        pipe_out = stage(target)
    else:
        stage()

    # Form a map from original qualname to gradient for equivalence check later
    pipe_grads = {}
    for origin_name, pipe_param in pipe_params.items():
        pipe_grads[origin_name] = pipe_param.grad

    # DDP reference model
    ref_mod = create_model()
    ref_mod.to(args.device)
    ddp_ref_mod = DistributedDataParallel(
        ref_mod,
        process_group=dp_group,
    )

    # DDP forward and backward
    ddp_out = ddp_ref_mod(input, target)
    ddp_out["loss"].backward()

    # Compare pipeline output and DDP output
    if pp_rank == args.pp_group_size - 1:
        torch.testing.assert_close(pipe_out, ddp_out)
        print("Output equivalence test passed")

    # Compare pipeline gradient and DDP gradient
    for origin_name, pipe_grad in pipe_grads.items():
        ddp_param = ddp_ref_mod.module.get_parameter(origin_name)
        if dp_rank == 0:
            print(f"Checking gradient of {origin_name}")
        # Since we use synthetic input and output, the gradients generated are
        # large. Hence we need to manually set relative tolerance
        torch.testing.assert_close(
            pipe_grad,
            ddp_param.grad,
            rtol=7e-2,
            atol=1e-5,
        )

    print("Gradient equivalence test passed")


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

    # pp group size must match with pipe_split's in model
    args.pp_group_size = 4
    # world size must be multiple of pp group size
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
