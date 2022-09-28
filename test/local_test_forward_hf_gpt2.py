# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
from typing import Dict

import torch
import torch.autograd.profiler_legacy
from transformers import GPT2Config, GPT2Model

import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer
from pippy.IR import (
    MultiUseParameterConfig,
    Pipe,
    PipeSplitWrapper,
    annotate_split_points,
)
from pippy.microbatch import TensorChunkSpec
from pippy.PipelineDriver import (
    PipelineDriver1F1B,
    PipelineDriverBase,
    PipelineDriverFillDrain,
)

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
}

VERBOSE = bool(int(os.environ.get("VERBOSE", False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

pippy.fx.Tracer.proxy_buffer_attributes = True


def run_master(_, args):
    bs = 20
    seq_length = 32

    MULTI_USE_PARAM_CONFIG = (
        MultiUseParameterConfig.REPLICATE
        if args.replicate
        else MultiUseParameterConfig.TRANSMIT
    )
    print(f"REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}")

    print("Using schedule:", args.schedule)

    gpt2 = GPT2Model(GPT2Config(use_cache=False))
    gpt2.to(args.device)
    gpt2.eval()
    gpt2_input = torch.zeros(
        bs, seq_length, dtype=torch.long, device=args.device
    ).random_(gpt2.config.vocab_size)

    for i in range(gpt2.config.n_layer):
        annotate_split_points(
            gpt2, {f"h.{i}": PipeSplitWrapper.SplitPoint.BEGINNING}
        )
    annotate_split_points(gpt2, {"ln_f": PipeSplitWrapper.SplitPoint.BEGINNING})

    input_names = gpt2.dummy_inputs.keys()
    sig = inspect.signature(gpt2.forward)
    concrete_args = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.name not in input_names
    }

    print("Instantiating GPT2 Pipeline")
    gpt2_pipe = Pipe.from_tracing(
        gpt2,
        MULTI_USE_PARAM_CONFIG,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
    )

    assert gpt2.config.n_layer + 2 == len(list(gpt2_pipe.split_gm.children()))

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec: Dict = {}
    output_chunk_spec = {"last_hidden_state": TensorChunkSpec(0)}

    pipe_driver: PipelineDriverBase = schedules[args.schedule](
        gpt2_pipe,
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
    print(
        "Running GPT2 pipeline. NB: if this is too slow, set OMP_NUM_THREADS to a higher value"
    )
    out = pipe_driver(gpt2_input)
    print("Running reference pipeline")
    ref_out = gpt2_pipe(gpt2_input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_close(
            out["last_hidden_state"], ref_out["last_hidden_state"]
        )
        print(
            f'equivalence test passed {torch.sum(out["last_hidden_state"])} ref {torch.sum(ref_out["last_hidden_state"])}'
        )

    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(
        enabled=PROFILING_ENABLED
    ) as prof:
        pipe_driver._debug_mask_minibatches = False
        pipe_driver.chunks = 5
        out = pipe_driver(gpt2_input)
        ref_out = gpt2_pipe(gpt2_input)
        print(
            f'profiling run completed {torch.sum(out["last_hidden_state"])} ref {torch.sum(ref_out["last_hidden_state"])}'
        )
    if PROFILING_ENABLED:
        prof.export_chrome_trace(
            f"{os.path.splitext(os.path.basename(__file__))[0]}.json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 14))
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
    args = parser.parse_args()

    run_pippy(run_master, args)
