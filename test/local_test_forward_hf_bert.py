# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import os

import pippy.fx

import torch
import torch.autograd.profiler_legacy
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer
from pippy.IR import (
    annotate_split_points,
    MultiUseParameterConfig,
    Pipe,
    PipeSplitWrapper,
)
from pippy.PipelineDriver import (
    PipelineDriver1F1B,
    PipelineDriverBase,
    PipelineDriverFillDrain,
)
from transformers import BertConfig, BertModel

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
}

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

    bert = BertModel(BertConfig())
    bert.to(args.device)
    bert.eval()
    bert_input = torch.zeros(
        bs, seq_length, dtype=torch.long, device=args.device
    ).random_(bert.config.vocab_size)
    bert(bert_input)

    for i in range(bert.config.num_hidden_layers):
        annotate_split_points(
            bert, {f"encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING}
        )
    annotate_split_points(
        bert, {"pooler": PipeSplitWrapper.SplitPoint.BEGINNING}
    )

    input_names = bert.dummy_inputs.keys()
    sig = inspect.signature(bert.forward)
    concrete_args = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.name not in input_names
    }

    print("Instantiating BERT Pipeline")
    bert_pipe = Pipe.from_tracing(
        bert,
        MULTI_USE_PARAM_CONFIG,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
    )

    assert bert.config.num_hidden_layers + 2 == len(
        list(bert_pipe.split_gm.children())
    )

    pipe_driver: PipelineDriverBase = schedules[args.schedule](
        bert_pipe,
        5,
        args.world_size,
        _debug_mask_minibatches=True,
        _record_mem_dumps=bool(args.record_mem_dumps),
        checkpoint=bool(args.checkpoint),
    )

    # # Warm up and correctness runs
    out = pipe_driver(bert_input)
    ref_out = bert_pipe(bert_input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_close(
            out["last_hidden_state"], ref_out["last_hidden_state"]
        )
        torch.testing.assert_close(
            out["pooler_output"], ref_out["pooler_output"]
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
        out = pipe_driver(bert_input)
        ref_out = bert_pipe(bert_input)
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
