# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import pippy.fx
import pippy.ModelSplit

import torch
import torch.distributed as dist
import torch.autograd.profiler_legacy
from pippy import run_pippy
from pippy.compile import compile_stage
from pippy.IR import MultiUseParameterConfig, Pipe
# from pippy.PipelineDriver import (
#     PipelineDriver1F1B,
#     PipelineDriverBase,
#     PipelineDriverFillDrain,
#     PipelineDriverInterleaved1F1B,
# )

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

# schedules = {
#     "FillDrain": PipelineDriverFillDrain,
#     "1F1B": PipelineDriver1F1B,
#     "Interleaved1F1B": PipelineDriverInterleaved1F1B,
# }

pippy.fx.Tracer.proxy_buffer_attributes = True

# MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.TRANSMIT


# Example model definition
d_hid = 512
bs = 503  # batch_size


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)
        self.register_buffer("buffer", torch.randn(bs, d_hid))

    def forward(self, x):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param) + self.buffer[: x.shape[0]]
        x = self.lin(x)
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        x = self.lin(x)
        x = torch.relu(x)

        return x


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


# Common function to run pipeline with input and check equivalence
def run_compile_stage(mod: torch.nn.Module, stage: Pipe, args):
    nstages = len(list(stage.split_gm.children()))
    ec_input = torch.randn(bs, d_hid, device=args.device)
    # Warm up and correctness runs

    if args.rank == 0:
        stage(ec_input)
    elif args.rank == args.world_size - 1:
        out = stage(ec_input)
    else:
        stage()

    print(f"Rank {args.rank} completes")

    # run with different chunk size to exercise microbatch and scheduling components
    # stage.chunks = 1
    # stage(ec_input)
    # stage.chunks = 100
    # stage(ec_input)

    # check numeric equivalence in the last rank
    if args.rank == args.world_size - 1:
        if CHECK_NUMERIC_EQUIVALENCE:
            ref_out = mod(ec_input)
            torch.testing.assert_close(out, ref_out)
            print(
                f'equivalence test passed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}'
            )

    # # # Profiling runs
    # with torch.autograd.profiler_legacy.profile(
    #     enabled=PROFILING_ENABLED
    # ) as prof:
    #     pipe_driver.chunks = nstages
    #     out = pipe_driver(ec_input)
    #     ref_out = ec_pipe(ec_input)
    #     print(
    #         f'profiling run completed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}'
    #     )
    # if PROFILING_ENABLED:
    #     prof.export_chrome_trace(
    #         f"{os.path.splitext(os.path.basename(__file__))[0]}.json"
    #     )


def test_split_on_size_threshold(args):
    ec = ExampleCode()
    ec.to(args.device)

    # Auto-split based on size threshold
    threshold = 300000
    split_policy = pippy.ModelSplit.split_on_size_threshold(threshold)

    ec_input = torch.randn(bs, d_hid, device=args.device)
    stage = compile_stage(  # default multi_use_param_spec to REPLICATE'ing param(if used in multiple stages) across stages instead of TRANSMI'ing it
        ec,
        args.rank,
        args.world_size,
        args.chunks,
        args.device,
        None,
        [ec_input,],
        split_policy=split_policy
    )

    inspect_split_module(stage, expected_stages=5)
    run_compile_stage(ec, stage, args)

    dist.barrier()


def test_split_into_equal_size(args):
    ec = ExampleCode()
    ec.to(args.device)

    # Auto-split based on given number of stages
    nstages = 5
    split_policy = pippy.ModelSplit.split_into_equal_size(nstages)

    ec_input = torch.randn(bs, d_hid, device=args.device)
    stage = compile_stage(  # default multi_use_param_spec to REPLICATE'ing param(if used in multiple stages) across stages instead of TRANSMI'ing it
        ec,
        args.rank,
        args.world_size,
        args.chunks,
        args.device,
        None,
        [ec_input,],
        split_policy=split_policy
    )

    inspect_split_module(stage, expected_stages=nstages)
    run_compile_stage(ec, stage, args)

    dist.barrier()



def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 5))
    )
    parser.add_argument(
        "--rank", type=int, default=int(os.getenv("RANK", -1))
    )
    parser.add_argument(
        "--chunks", type=int, default=4,
    )
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
        "--record_mem_dumps", type=int, default=0, choices=[0, 1]
    )
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    args = parser.parse_args(args)

    args.device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}") \
                    if args.cuda else torch.device("cpu")

    # init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    test_split_into_equal_size(args)
    # test_split_on_size_threshold(args)


if __name__ == "__main__":
    main()


class LocalTestAutoSplit(unittest.TestCase):
    def test_auto_split(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
