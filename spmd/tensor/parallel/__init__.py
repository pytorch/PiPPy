# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)

from spmd.tensor.parallel.api import (
    parallelize_module,
)

from spmd.tensor.parallel.style import (
    ParallelStyle,
    PairwiseParallel,
    RowwiseParallel,
    ColwiseParallel,
    make_input_shard_1d,
    make_input_replicate_1d,
    make_output_shard_1d,
    make_output_replicate_1d,
    make_output_tensor,
)

__all__ = [
    "TensorParallelMultiheadAttention",
    "parallelize_module",
    "ParallelStyle",
    "PairwiseParallel",
    "RowwiseParallel",
    "ColwiseParallel",
    "make_input_shard_1d",
    "make_input_replicate_1d",
    "make_output_shard_1d",
    "make_output_replicate_1d",
    "make_output_tensor",
]
