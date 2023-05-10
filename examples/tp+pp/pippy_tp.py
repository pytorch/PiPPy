# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch

import pippy
from pippy.PipelineStage import compile_stage
import pippy.fx
from pippy.IR import pipe_split

import torch.distributed as dist
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


d_hid = 256
batch_size_per_chunk = 8


def run_all(args):
    # The seed here has two purposes:
    # - Ensure all TP ranks have same input
    # - Ensure the model (ec) created are the same, as if it comes from a
    # single, big model before partitioning
    torch.manual_seed(0)

    # Create original model
    ec = ExampleCode(d_hid)
    ec.to(args.device)

    # Create input
    chunks = args.pp_group_size
    inp_size = [chunks * batch_size_per_chunk, d_hid]
    device_type = args.device.type
    inp = torch.rand(*inp_size, device=device_type)

    # Create global DeviceMesh
    ranks = torch.arange(args.world_size)
    rank_mesh = ranks.reshape(args.pp_group_size, args.tp_group_size)
    pp_dim = 0
    tp_dim = 1
    dm = DeviceMesh(
        device_type,
        rank_mesh,
    )

    # Figure out my PP and TP rank
    pp_rank = args.rank // args.tp_group_size
    tp_rank = args.rank % args.tp_group_size
    print(f"Global rank {args.rank}, pp rank: {pp_rank}, tp rank: {tp_rank}")

    # Get pp group
    # `tp_rank` can serve as pipeline id
    print(f"Rank {args.rank} Instantiating pipeline with ranks {dm.mesh[:, tp_rank]}")
    pp_group = dm.get_dim_groups()[pp_dim]

    # Get stage module (on all pp ranks)
    stage = compile_stage(
        ec,
        pp_rank,
        args.pp_group_size,
        args.device,
        group=pp_group,
        example_input=inp,
    )

    # Tensor parallelize submodules
    print(f"Rank {args.rank} TP-lize submodule with {dm.mesh[pp_rank]}")
    parallelize_module(stage.submod, dm, PairwiseParallel(), tp_mesh_dim = tp_dim)

    if pp_rank == 0:
        out = stage(inp)
    elif pp_rank == args.pp_group_size - 1:
        out = stage()
    else:
        stage()

    # Last rank checks result
    if pp_rank == args.pp_group_size - 1:
        print(f"Pipeline {tp_rank} output: {out}")


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
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    args = parser.parse_args(args)

    # Use world size to determine TP group size
    assert args.world_size % args.pp_group_size == 0
    args.tp_group_size = args.world_size // args.pp_group_size
    if args.rank == 0:
        print(
            f"Pipeline parallel size: {args.pp_group_size}\n"
            f"Tensor parallel size: {args.tp_group_size}"
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

    run_all(args)


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
