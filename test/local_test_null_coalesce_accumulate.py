# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch

import pippy.fx
from pippy import run_pippy
from pippy.IR import (
    Pipe,
    TrivialLossWrapper,
    pipe_split,
    _null_coalesce_accumulate,
)
from pippy.PipelineDriver import (
    PipelineDriverBase,
    PipelineDriverFillDrain,
    PipelineDriver1F1B,
)
from pippy.microbatch import TensorChunkSpec, CustomReducer

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
}

pippy.fx.Tracer.proxy_buffer_attributes = True


def run_master(_, args):
    all_ranks = list(range(1, args.world_size))  # exclude master rank = 0
    chunks = len(all_ranks)
    bs = 4 * chunks
    hid_dim = 50

    class Code(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(hid_dim, hid_dim)

        def forward(self, x):
            x = self.linear(x)
            pipe_split()
            y = torch.relu(x)
            pipe_split()
            z = torch.sigmoid(x)
            pipe_split()
            return y + z

    c = Code()
    c.train()
    mse_loss = torch.nn.MSELoss()
    wrapper = TrivialLossWrapper(c, mse_loss)
    accum_pipe = Pipe.from_tracing(wrapper)
    assert 4 == len(list(accum_pipe.split_gm.children()))
    assert any(
        n.target == _null_coalesce_accumulate
        for n in accum_pipe.split_gm.graph.nodes
    )

    input = torch.randn(bs, hid_dim)
    target = torch.randn(bs, hid_dim)
    accum_pipe(input, target)

    output_chunk_spec = CustomReducer(torch.tensor(0.0), lambda a, b: a + b)
    pipe_driver: PipelineDriverBase = schedules[args.schedule](
        accum_pipe,
        chunks,
        output_chunk_spec,
        args.world_size - 1,
        all_ranks=all_ranks,
        _debug_mask_minibatches=True,
        _record_mem_dumps=bool(args.record_mem_dumps),
        checkpoint=bool(args.checkpoint),
    )

    pipe_driver(input, target)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 5))
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

    run_pippy(run_master, args)


if __name__ == "__main__":
    main()


class LocalTestNullCoalesceAccumulateTest(unittest.TestCase):
    def test_null_coalesce_accumulate(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--cuda",
            os.getenv("USE_CUDA", "0"),
            "--master_port",
            str(port),
        ]
        main(args)
