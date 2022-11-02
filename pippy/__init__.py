# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
from pippy.IR import (
    PipeSequential,
    LossWrapper,
    TrivialLossWrapper,
    pipe_split,
    Pipe,
    PipeSplitWrapper,
    annotate_split_points,
)
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B
from pippy.ModelSplit import split_on_size_threshold, split_into_equal_size
from pippy.utils import run_pippy


__all__ = [
    "PipeSequential",
    "LossWrapper",
    "TrivialLossWrapper",
    "Pipe",
    "pipe_split",
    "run_pippy",
    "PipeSplitWrapper",
    "annotate_split_points",
    "PipelineDriverFillDrain",
    "PipelineDriver1F1B",
    "split_into_equal_size",
    "split_on_size_threshold",
]

pippy_debug = os.environ.get("PIPPY_DEBUG", "OFF")

if pippy_debug == "INFO":
    logging.getLogger().setLevel(logging.INFO)
elif pippy_debug == "DEBUG":
    logging.getLogger().setLevel(logging.DEBUG)
