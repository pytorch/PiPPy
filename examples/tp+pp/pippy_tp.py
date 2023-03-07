# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch

import pippy
import pippy.fx
from pippy import run_pippy
from pippy.IR import pipe_split

from torch.distributed._tensor import (
    DeviceMesh,
)
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)


pippy.fx.Tracer.proxy_buffer_attributes = True


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

    def forward(self, x):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        return x


def run_all(pp_ranks, args):
    chunks = args.pp_group_size
    device_type = "cuda" if args.cuda else "cpu"

    # Figure out my PP rank
    pp_rank = args.rank // args.tp_group_size
    print(f"Global rank {args.rank}, pipeline: {pp_ranks}, my rank in pipe: {pp_rank}")

    d_hid = 256
    batch_size_per_chunk = 8
    inp_size = [chunks * batch_size_per_chunk, d_hid]

    # Ensure all tp ranks have same input.
    torch.manual_seed(0)
    inp = torch.rand(*inp_size, device=device_type)

    """
    # Reference run
    # `torchrun --nproc_per_node=2 pippy_tp_mlp.py --world_size=2 --pp_group_size=1`
    ec_tp = ExampleCode(d_hid)
    ec_tp.to(args.device)

    start_idx = 0
    device_mesh = DeviceMesh(
        device_type,
        list(range(start_idx, start_idx + args.tp_group_size)),
    )
    print(f"Rank {args.rank} calling parallelize_module with {device_mesh}")
    parallelize_module(ec_tp, device_mesh, PairwiseParallel())
    print(f"Rank {args.rank} sharding complete")

    ref_out = ec_tp(inp)
    print(f"Ref out: {ref_out.size()}")
    """

    # PiPPy run
    ec = ExampleCode(d_hid)
    ec.to(args.device)

    # Get:
    # - pipeline driver (for pipeline head rank)
    # - stage submodule (for all ranks)
    pipe_driver, submod = pippy.all_compile(
        ec,
        args.pp_group_size,
        chunks,
        ranks=pp_ranks,
    )

    # Create TP device mesh
    my_device_mesh = None
    for stage in range(args.pp_group_size):
        start_rank = stage * args.tp_group_size
        tp_ranks = list(range(start_rank, start_rank + args.tp_group_size))
        tp_device_mesh = DeviceMesh(
            device_type,
            tp_ranks,
        )
        if stage == pp_rank:
            my_device_mesh = tp_device_mesh

    # Tensor parallelize submodules
    print(f"Rank {args.rank} calling parallelize_module with {my_device_mesh}")
    parallelize_module(submod, my_device_mesh, PairwiseParallel())

    if pp_rank == 0:
        print(f"Rank {args.rank} Instantiated pipeline with ranks {pp_ranks}")
        out = pipe_driver(inp)
        print(f"Pipeline {args.rank} output: {out.size()}")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 8))
    )
    # ExampleCode has two stages
    parser.add_argument(
        "--pp_group_size", type=int, default=4,
    )
    # in row-major
    # TP ranks are contiguous rows of size `args.tp_group_size`
    # PP ranks are non-contiguous columns of size `args.pp_group_size`
    #
    # if tp_group_size = 4 and pp_group_size = 3
    #
    #   0 1 2  3
    #   4 5 6  7
    #   8 9 10 11
    #
    # TP ranks are [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    # PP ranks are [0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]
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

    # Use world size to determine TP group size
    assert args.world_size % args.pp_group_size == 0
    args.tp_group_size = args.world_size // args.pp_group_size
    print(f"Using tensor parallel group size: {args.tp_group_size}")

    # All ranks participate
    args.gspmd = 1
    run_pippy(run_all, args)


if __name__ == "__main__":
    main()


class LocalTestPiPPyTP(unittest.TestCase):
    def test_pp_tp(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
