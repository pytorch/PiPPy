# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)

from spmd.tensor.parallel.api import (
    tp_shard_self_attn,
    replicate_input,
    replicate_output,
    _parallelize_linear,
)

from spmd.tensor.parallel.style import (
    ParallelStyle,
    RowwiseParallel,
    ColwiseParallel,
    PairwiseParallel,
)
