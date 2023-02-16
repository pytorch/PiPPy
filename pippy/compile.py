# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Any, Callable, List, Optional
from pippy.PipelineDriver import PipelineDriver1F1B, PipelineDriverFillDrain, PipelineDriverInterleaved1F1B
import pippy.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe
from pippy.microbatch import LossReducer, sum_reducer

import torch


PIPELINE_SCHEDULE_DRIVERS = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}

def compile(
    mod: torch.nn.Module,
    num_ranks: int,
    num_chunks: int,
    schedule: Optional[str] = "FillDrain",
    split_policy: Optional[
        Callable[[fx.GraphModule], fx.GraphModule]
    ] = None,
    rank: int = None,
    ranks: List[int] = None,
    tracer=None,
    loss_reducer: LossReducer = sum_reducer,
    args_chunk_spec=None,
    kwargs_chunk_spec=None,
    output_chunk_spec=None,
    checkpoint=False,
    _debug_mask_minibatches: bool = False,
    **kwargs,
):
    # If a param will be used in multiple pipeline stages, we default the strategy to REPLICATE'ing the param across
    # stages instead of TRANSMIT'ting it
    multi_use_param_spec = MultiUseParameterConfig.REPLICATE

    # Figure out which output is loss from output_chunk_spec
    output_loss_value_spec: Any = None
    if isinstance(output_chunk_spec, dict):
        output_loss_value_spec = {
            k: isinstance(v, LossReducer) for k, v in output_chunk_spec.items()
        }

    logging.info("[PiPPy] Splitting model ...")
    pipe_model = Pipe.from_tracing(
        mod,
        multi_use_param_spec=multi_use_param_spec,
        tracer=tracer,
        output_loss_value_spec=output_loss_value_spec,
        split_policy=split_policy,
        **kwargs,
    )
    logging.info(pipe_model.split_gm)

    logging.info("[PiPPy] Creating pipeline driver ...")
    if schedule not in PIPELINE_SCHEDULE_DRIVERS:
        raise ValueError(
            f"Unknown pipeline schedule: {schedule}. "
            f"Please select from {PIPELINE_SCHEDULE_DRIVERS.keys()}"
        )
    pipeline_driver = PIPELINE_SCHEDULE_DRIVERS[schedule](
        pipe_model,
        num_chunks,
        num_ranks,
        all_ranks=ranks,
        args_chunk_spec=args_chunk_spec,
        kwargs_chunk_spec=kwargs_chunk_spec,
        output_chunk_spec=output_chunk_spec,
        checkpoint=checkpoint,
        loss_reducer=loss_reducer,
        _debug_mask_minibatches=_debug_mask_minibatches,
    )

    return pipeline_driver
