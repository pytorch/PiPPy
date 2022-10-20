# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)

from spmd.tensor.parallel.api import (
    tp_shard_self_attn,
    tp_shard_mlp,
    replicate_input,
    replicate_output,
)
