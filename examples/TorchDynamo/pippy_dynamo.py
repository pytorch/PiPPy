# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest
from typing import Dict

import torch
import torch.autograd.profiler_legacy
import torch.fx
# TorchDynamo is moved into PyTorch for PyTorch > 1.13
# One can use TorchDynamo without installing it separately by:
# import torch._dynamo as dynamo
# If your PyTorch version is <= 1.13.0, please install torchdynamo separately
import torchdynamo as dynamo

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

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}

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
    # PiPPy parameters
    MULTI_USE_PARAM_CONFIG = (
        MultiUseParameterConfig.REPLICATE
        if args.replicate
        else MultiUseParameterConfig.TRANSMIT
    )
    print(f"REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}")
    print("Using schedule:", args.schedule)

    # Model parameters
    d_hid = 512
    bs = 503

    # Chunking parameters
    output_chunk_spec = (TensorChunkSpec(0),)
    chunks = 1

    # Ask Dynamo to let PiPPy annotation stay in graph
    dynamo.allow_in_graph(pipe_split)

    # Define a compiler backend made by PiPPy for use by Dynamo
    # The backend comprising:
    # - PiPPy's graph split, and
    # - PiPPy's driver creation
    # The driver is return as a compiled runtime callable, which will be used in the actual data execution
    def my_pippy_compiler(gm: torch.fx.GraphModule, example_inputs, **kwargs):
        print("\n============= my_pippy_compiler() called with FX graph =============")
        gm.graph.print_tabular()

        # PiPPy Pipe creation
        # Here we split the graph
        pipe = Pipe.from_tracing(gm, MULTI_USE_PARAM_CONFIG)
        inspect_split_module(pipe, 4)

        # Create PipelineDriver
        pipe_driver: PipelineDriverBase = schedules[args.schedule](
            pipe,
            chunks,
            output_chunk_spec,
            args.world_size,
            checkpoint=bool(args.checkpoint),
            use_c10d=True,
        )

        # Return a runtime Callable
        # This PipelineDriver is a distributed runtime
        return pipe_driver

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.lin = torch.nn.Linear(d_hid, d_hid)
            self.register_buffer("buffer", torch.randn(bs + 100, d_hid))

        # Decorate with Dynamo here, or
        # explicitly call optimize in the main code.
        # We do the latter for zero change on the model, hence commenting out the decoration here
        # @dynamo.optimize(my_pippy_compiler)
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
            return x

    # Create model as usual
    ec = ExampleCode()
    ec.to(args.device)
    ec_input = torch.randn(bs, d_hid, device=args.device)
    ref_out = ec(ec_input)

    # Optimize and distribute model using Dynamo + PiPPy
    ec_pipe = dynamo.optimize(my_pippy_compiler)(ec)

    print(f"\n======= Runtime tests =======")
    # This would already be output returned by PiPPy's distributed pipeline
    pipe_out = ec_pipe(ec_input)

    # Check correctness
    torch.testing.assert_close(pipe_out, ref_out)
    print(
        f'equivalence test passed {torch.sum(pipe_out)} ref {torch.sum(ref_out)}'
    )

    # Profiling run
    # This run would not trigger compilation
    # We can also change the size to test dynamic shape support
    # ec_input = torch.randn(bs + 10, d_hid, device=args.device)
    with torch.autograd.profiler_legacy.profile(
            enabled=PROFILING_ENABLED
    ) as prof:
        pipe_out = ec_pipe(ec_input)
        print(
            f'profiling run completed {torch.sum(pipe_out)} ref {torch.sum(ref_out)}'
        )
    if PROFILING_ENABLED:
        prof.export_chrome_trace(
            f"{os.path.splitext(os.path.basename(__file__))[0]}.json"
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
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    args = parser.parse_args(args)

    # Interleaved 1F1B uses fewer ranks than number of stages
    if args.schedule == "Interleaved1F1B":
        args.world_size = 2

    run_pippy(run_master, args)


if __name__ == "__main__":
    main()


class LocalTestDynamoTest(unittest.TestCase):
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
