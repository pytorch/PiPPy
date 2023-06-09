# Copyright (c) Meta Platforms, Inc. and affiliates
import inspect
import logging
from typing import Any, Callable, List, Optional
from pippy.PipelineDriver import (
    PipelineDriver1F1B,
    PipelineDriverFillDrain,
    PipelineDriverInterleaved1F1B,
)
from pippy.PipelineStage import PipelineStage
import pippy.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, PiPPyShapeProp
from pippy.microbatch import (
    LossReducer,
    gen_output_chunk_spec,
    split_args_kwargs_into_chunks,
    sum_reducer,
)
from pippy.utils import get_device, get_pp_rank, get_rank
from pippy.debug import PIPPY_VERBOSITY

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode


PIPELINE_SCHEDULE_DRIVERS = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}


def create_default_args(
    mod: torch.nn.Module,
    except_keys: List = None,
):
    if except_keys is None:
        except_keys = []
    sig = inspect.signature(mod.forward)
    default_kwargs = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.name not in except_keys and p.default is not inspect._empty
    }
    return default_kwargs


def _compile(
    all_compile: bool,
    mod: torch.nn.Module,
    num_ranks: int,
    num_chunks: int,
    schedule: Optional[str] = "FillDrain",
    split_policy: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
    ranks: List[int] = None,
    tracer=None,
    loss_reducer: LossReducer = sum_reducer,
    args_chunk_spec=None,
    kwargs_chunk_spec=None,
    output_chunk_spec=None,
    checkpoint=False,
    _debug_mask_minibatches: bool = False,
    index_filename=None,
    checkpoint_prefix: str = None,
    **kwargs,
):
    if ranks is None:
        ranks = list(range(num_ranks))

    if all_compile:
        rank = get_rank()
        pp_rank = get_pp_rank(rank, ranks)
    else:
        pp_rank = 0

    # If a param will be used in multiple pipeline stages, we default the strategy to REPLICATE'ing the param across
    # stages instead of TRANSMIT'ting it
    multi_use_param_spec = MultiUseParameterConfig.REPLICATE

    # Figure out which output is loss from output_chunk_spec
    output_loss_value_spec: Any = None
    if isinstance(output_chunk_spec, dict):
        output_loss_value_spec = {
            k: isinstance(v, LossReducer) for k, v in output_chunk_spec.items()
        }

    logging.info("[PiPPy] Tracing model ...")
    pipe_model = Pipe.from_tracing(
        mod,
        multi_use_param_spec=multi_use_param_spec,
        tracer=tracer,
        output_loss_value_spec=output_loss_value_spec,
        split_policy=split_policy,
        **kwargs,
    )

    # In all_compile mode, each rank calls pippy.all_compile, hence they will all have the pipe.
    # We can hence ask each rank to get its own stage from the pipe, and materialize it locally.
    if all_compile:
        device = get_device()

        # `None` means self.dtype, i.e. no change
        dtype = None
        # TODO: generalize this
        if hasattr(mod, "config") and hasattr(mod.config, "torch_dtype"):
            dtype = mod.config.torch_dtype  # type: ignore[union-attr]

        pipe_model.defer_stage_init(
            device,
            index_filename,
            dtype,
            checkpoint_prefix,
        )
        stage_mod = pipe_model.export(pp_rank)

    if pp_rank == 0:
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

    if not all_compile:
        return pipeline_driver

    if pp_rank == 0:
        return pipeline_driver, stage_mod
    else:
        return None, stage_mod


def compile(
    mod: torch.nn.Module,
    num_ranks: int,
    num_chunks: int,
    schedule: Optional[str] = "FillDrain",
    split_policy: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
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
    return _compile(
        False,
        mod,
        num_ranks,
        num_chunks,
        schedule=schedule,
        split_policy=split_policy,
        ranks=ranks,
        tracer=tracer,
        loss_reducer=loss_reducer,
        args_chunk_spec=args_chunk_spec,
        kwargs_chunk_spec=kwargs_chunk_spec,
        output_chunk_spec=output_chunk_spec,
        checkpoint=checkpoint,
        _debug_mask_minibatches=_debug_mask_minibatches,
        **kwargs,
    )


def all_compile(
    mod: torch.nn.Module,
    num_ranks: int,
    num_chunks: int,
    schedule: Optional[str] = "FillDrain",
    split_policy: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
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
    return _compile(
        True,
        mod,
        num_ranks,
        num_chunks,
        schedule=schedule,
        split_policy=split_policy,
        ranks=ranks,
        tracer=tracer,
        loss_reducer=loss_reducer,
        args_chunk_spec=args_chunk_spec,
        kwargs_chunk_spec=kwargs_chunk_spec,
        output_chunk_spec=output_chunk_spec,
        checkpoint=checkpoint,
        _debug_mask_minibatches=_debug_mask_minibatches,
        **kwargs,
    )


def compile_stage(
    mod: torch.nn.Module,
    rank: int,
    num_ranks: int,
    num_chunks: int,
    device: torch.device,
    group: dist.ProcessGroup,
    example_inputs: List[torch.Tensor],
    split_policy: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
    return_to_0: bool = False,
    tracer=None,
    loss_reducer: LossReducer = sum_reducer,
    args_chunk_spec=None,
    kwargs_chunk_spec=None,
    output_chunk_spec=None,
    **kwargs,
) -> PipelineStage:
    # If a param will be used in multiple pipeline stages, we default the strategy to REPLICATE'ing the param across
    # stages instead of TRANSMIT'ting it
    multi_use_param_spec = MultiUseParameterConfig.REPLICATE

    # Figure out which output is loss from output_chunk_spec
    output_loss_value_spec: Any = None
    if isinstance(output_chunk_spec, dict):
        output_loss_value_spec = {
            k: isinstance(v, LossReducer) for k, v in output_chunk_spec.items()
        }

    logging.info("[PiPPy] Tracing model ...")
    pipe = Pipe.from_tracing(
        mod,
        multi_use_param_spec=multi_use_param_spec,
        tracer=tracer,
        output_loss_value_spec=output_loss_value_spec,
        split_policy=split_policy,
        **kwargs,
    )

    gm = pipe.split_gm
    if rank == 0:
        logging.info(gm)
        if PIPPY_VERBOSITY == "INFO":
            gm.graph.print_tabular()

    # Get shape of chunked arguments
    args_split, _ = split_args_kwargs_into_chunks(
        example_inputs,
        {},  # kwargs included in `example_inputs`
        num_chunks,
        args_chunk_spec,
        kwargs_chunk_spec,  # TODO: merge into args_chunk_spec
    )

    # Use fake tensor for shape propagation
    # Since model itself may have been materialized, we need to use
    # `allow_non_fake_inputs`
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    # In reality, the fake input should be created from shape info (potentially
    # broadcast from Rank 0)
    fake_args_split = fx.node.map_aggregate(
        args_split, lambda a: fake_mode.from_tensor(a)
    )

    # Use 1st chunk of args for shape propagation
    chunk0 = fake_args_split[0]

    sp = PiPPyShapeProp(gm)
    sp.propagate(*chunk0)

    # Prepare output chunk/reduce spec for merging/reducing final outputs
    output_chunk_spec = (
        output_chunk_spec
        if output_chunk_spec
        else gen_output_chunk_spec(pipe.loss_spec, loss_reducer)
    )

    # Create pipeline stage
    return PipelineStage(
        pipe,
        rank,
        num_ranks,
        num_chunks,
        device,
        group,
        return_to_0,
        args_chunk_spec,
        kwargs_chunk_spec,
        output_chunk_spec,
    )
