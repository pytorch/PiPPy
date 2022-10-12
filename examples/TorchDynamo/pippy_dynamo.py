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

import torchdynamo

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get("VERBOSE", False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

pippy.fx.Tracer.proxy_buffer_attributes = True


def inspect_split_module(
    pipe: Pipe,
    expected_stages: int = -1,
):
    gm: pippy.fx.GraphModule = pipe.split_gm
    # Check returned number of stages
    nstages = len(list(gm.children()))
    if expected_stages > 0:
        assert (
            nstages == expected_stages
        ), f"Model is split into {nstages} instead of {expected_stages} stages"

    print(f"\n======= GraphModule after Auto-split =======")
    print(gm)

    for i, submod in enumerate(gm.children()):
        print(f"\n======= Child module {i} =======")
        print(submod)


def run_master(_, args):
    d_hid = 512
    bs = 503

    MULTI_USE_PARAM_CONFIG = (
        MultiUseParameterConfig.REPLICATE
        if args.replicate
        else MultiUseParameterConfig.TRANSMIT
    )
    print(f"REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}")
    print("Using schedule:", args.schedule)

    def my_compiler(gm: torch.fx.GraphModule, example_inputs):
        print("\n============= my_compiler() called with FX graph =============")
        gm.graph.print_tabular()

        print("\n============= example inputs =============")
        print(example_inputs)

        # Run example input
        input = example_inputs[0]
        output = gm(input)
        print("\n============= example output =============")
        print(output)

        # PiPPy Pipe creation
        pipe = Pipe.from_tracing(gm, MULTI_USE_PARAM_CONFIG)
        print("\n============= PiPPy Pipe creation =============")

        inspect_split_module(pipe, 2)

        return pipe   # return a runtime Callable

    class ExampleCode(torch.nn.Module):
        @torchdynamo.optimize(my_compiler)
        def forward(self, a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            pipe_split()
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

    torchdynamo.allow_in_graph(pipe_split)
    ec = ExampleCode()
    ec(torch.randn(10))
    ec(torch.randn(10))

    """
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
            x = self.lin(x)
            pipe_split()
            x = torch.relu(x)
            return {"out": x}
    """

    """
    ec = ExampleCode()
    ec.to(args.device)
    ec_input = torch.randn(bs, d_hid, device=args.device)
    ec(ec_input)

    ec_pipe = Pipe.from_tracing(ec, MULTI_USE_PARAM_CONFIG)
    print(ec_pipe.split_gm)

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec: Dict = {}
    output_chunk_spec = {"out": TensorChunkSpec(0)}

    pipe_driver: PipelineDriverBase = schedules[args.schedule](
        ec_pipe,
        5,
        args_chunk_spec,
        kwargs_chunk_spec,
        output_chunk_spec,
        args.world_size,
        _debug_mask_minibatches=True,
        _record_mem_dumps=bool(args.record_mem_dumps),
        checkpoint=bool(args.checkpoint),
    )

    # # Warm up and correctness runs
    out = pipe_driver(ec_input)
    ref_out = ec_pipe(ec_input)

    # run with different chunk size to exercise microbatch and scheduling components
    pipe_driver.chunks = 1
    pipe_driver(ec_input)
    pipe_driver.chunks = 100
    pipe_driver(ec_input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_close(out["out"], ref_out["out"])
        print(
            f'equivalence test passed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}'
        )

    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(
        enabled=PROFILING_ENABLED
    ) as prof:
        pipe_driver.chunks = 5
        out = pipe_driver(ec_input)
        ref_out = ec_pipe(ec_input)
        print(
            f'profiling run completed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}'
        )
    if PROFILING_ENABLED:
        prof.export_chrome_trace(
            f"{os.path.splitext(os.path.basename(__file__))[0]}.json"
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

    # run_pippy(run_master, args)
    run_master(0, args)


if __name__ == "__main__":
    main()


class LocalTestForwardTest(unittest.TestCase):
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
