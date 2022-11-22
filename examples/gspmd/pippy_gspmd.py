# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import logging
import os
import unittest
from typing import Dict

import torch
import torch.autograd.profiler_legacy

import pippy.fx
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split
from pippy.PipelineDriver import (
    PipelineDriverBase,
    PipelineDriverFillDrain,
    PipelineDriver1F1B,
    PipelineDriverInterleaved1F1B,
)
from pippy.microbatch import TensorChunkSpec

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get("VERBOSE", False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

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
    MULTI_USE_PARAM_CONFIG = (
        MultiUseParameterConfig.REPLICATE
        if args.replicate
        else MultiUseParameterConfig.TRANSMIT
    )

    # Make sure all ranks have the same seed
    torch.manual_seed(5)
    ec = ExampleCode()

    ec_pipe = Pipe.from_tracing(ec, MULTI_USE_PARAM_CONFIG)
    ec_pipe.defer_stage_init(args.device)

    # Make sure every rank has deferred its stage init before master creates the driver
    pippy.utils.pp_group_barrier()

    if args.rank > 0:
        return  # Workers stop here

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec: Dict = {}
    output_chunk_spec = {"out": TensorChunkSpec(0)}
    chunks = 5

    pipe_driver: PipelineDriverBase = schedules[args.schedule](
        ec_pipe,
        chunks,
        args_chunk_spec,
        kwargs_chunk_spec,
        output_chunk_spec,
        args.world_size,
        _debug_mask_minibatches=True,
        _record_mem_dumps=bool(args.record_mem_dumps),
        checkpoint=bool(args.checkpoint),
    )

    # # Warm up and correctness runs
    ec_input = torch.randn(bs, d_hid, device=args.device)
    out = pipe_driver(ec_input)

    ec.to(args.device)
    ref_out = ec(ec_input)

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
        default=list(schedules.keys())[0],
        choices=schedules.keys(),
    )
    parser.add_argument(
        "--replicate", type=int, default=int(os.getenv("REPLICATE", "0"))
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument(
        "--record_mem_dumps", type=int, default=0, choices=[0, 1]
    )
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    args = parser.parse_args(args)

    # Interleaved 1F1B uses less ranks than number of stages
    if args.schedule == "Interleaved1F1B":
        args.world_size = 2

    args.gspmd = 1
    run_pippy(run_gspmd, args)


if __name__ == "__main__":
    main()


class LocalTestGspmdTest(unittest.TestCase):
    def test_forward(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--cuda",
            os.getenv("USE_CUDA", "0"),
            "--master_port",
            str(port),
        ]
        main(args)
