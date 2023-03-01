# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import copy
import functools
import logging
import os
import unittest

import torch
import torch.nn as nn

import pippy.fx
from pippy import run_pippy
from pippy.IR import (
    MultiUseParameterConfig,
    Pipe,
    pipe_split,
)
from pippy.PipelineDriver import (
    PipelineDriverFillDrain,
    PipelineDriver1F1B,
    PipelineDriverBase,
    PipelineDriverInterleaved1F1B,
)

from spmd import (
    distribute_tensor,
    distribute_module,
    DeviceMesh,
    DTensor,
    Shard,
    Replicate,
)

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get("VERBOSE", False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.INFO)

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

    def forward(self, x):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        return x


def _gradient_hook(param, grad):
    param._local_tensor.grad = grad._local_tensor


def shard_mlp(m, device_mesh):
    print(f"Calling shard_mlp with {device_mesh}")
    col_wise_sharding = [Shard(0)]
    row_wise_sharding = [Shard(1)]
    replicate = [Replicate()] * device_mesh.ndim

    def shard_params(name, module):
        if isinstance(module, nn.Linear):
            if name == "net1":
                print(f"shard_mlp: sharding {name}")
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, col_wise_sharding
                    )
                )
                sharded_bias = nn.Parameter(
                    distribute_tensor(
                        module.bias, device_mesh, col_wise_sharding
                    )
                )
                module.register_parameter("weight", sharded_weight)
                module.register_parameter("bias", sharded_bias)
                module.weight.register_hook(
                    functools.partial(_gradient_hook, module.weight)
                )
            elif name == "net2":
                print(f"shard_mlp: sharding {name}")
                sharded_weight = nn.Parameter(
                    distribute_tensor(
                        module.weight, device_mesh, row_wise_sharding
                    )
                )
                replicated_bias = nn.Parameter(
                    distribute_tensor(module.bias, device_mesh, replicate)
                )
                module.register_parameter("weight", sharded_weight)
                module.register_parameter("bias", replicated_bias)

    def replicate_input(inputs):
        return DTensor.from_local(inputs[0], device_mesh, replicate)

    def aggregate_output(outputs):
        assert isinstance(outputs, DTensor)
        return (
            outputs.redistribute(outputs.device_mesh, replicate)
            .contiguous()
            .to_local()
        )

    dist_mod = distribute_module(
        m,
        device_mesh,
        partition_fn=shard_params,
        input_fn=replicate_input,
        output_fn=aggregate_output,
    )
    return dist_mod


def run_gspmd(pp_ranks, args):
    CHUNKS = 1
    print("Using schedule:", args.schedule)
    print(f"Master rank {args.rank}, PP ranks: {pp_ranks}")

    device_type = "cuda" if args.cuda else "cpu"

    d_hid = 10
    inp_size = [5, d_hid]

    # Ensure all tp ranks have same input.
    torch.manual_seed(0)
    inp = torch.rand(*inp_size, device=device_type)

    """
    # Reference run
    ec_tp = ExampleCode(d_hid)
    ec_tp.to(args.device)

    start_idx = 0
    device_mesh = DeviceMesh(
        device_type,
        list(range(start_idx, start_idx + args.tp_group_size)),
    )
    shard_mlp(ec_tp, device_mesh)
    print(f"Rank {args.rank} sharding complete")

    ref_out = ec_tp(inp)
    print(f"Ref out: {ref_out.size()}")
    """

    # PiPPy run
    ec = ExampleCode(d_hid)

    # PiPPy tracing
    ec_pipe = Pipe.from_tracing(ec)
    if args.rank == 0:
        print(ec_pipe.split_gm)

    # Figure out my PP rank
    pp_rank = 0
    for r in pp_ranks:
        if r == args.rank:
            break
        pp_rank += 1

    # Create TP device mesh
    start_rank = pp_rank * args.tp_group_size
    tp_ranks = list(range(start_rank, start_rank + args.tp_group_size))
    tp_device_mesh = DeviceMesh(
        device_type,
        tp_ranks,
    )

    # Tensor parallelize submodules
    submod = ec_pipe.export(stage_id=pp_rank)
    shard_mlp(submod, tp_device_mesh)

    # Make sure every rank has tp'lize its stage before master creates the driver
    pippy.utils.pp_group_barrier()

    if args.rank >= args.tp_group_size:
        return  # Non-masters stop here

    # Pipeline master code
    pipe_driver: PipelineDriverBase = schedules[args.schedule](
        ec_pipe,
        CHUNKS,
        args.pp_group_size,
        all_ranks=pp_ranks,
    )
    print(f"Rank {args.rank} Instantiated pipe with ranks {pp_ranks}")

    out = pipe_driver(inp)
    print(f"PiPPy out: {out.size()}")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
    # ExampleCode has two stages
    parser.add_argument(
        "--pp_group_size", type=int, default=2,
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
        default=list(schedules.keys())[0],
        choices=schedules.keys(),
    )
    parser.add_argument(
        "--replicate", type=int, default=int(os.getenv("REPLICATE", "0"))
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    args = parser.parse_args(args)

    # Use world size to determine TP group size
    assert args.world_size % args.pp_group_size == 0
    args.tp_group_size = args.world_size // args.pp_group_size
    print(f"Using tensor parallel group size: {args.tp_group_size}")

    args.gspmd = 1
    run_pippy(run_gspmd, args)


if __name__ == "__main__":
    main()


class LocalTestPiPPyTP(unittest.TestCase):
    def test_ddp(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
