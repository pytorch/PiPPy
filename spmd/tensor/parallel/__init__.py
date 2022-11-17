# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.parallel.multihead_attention_tp import (
    TensorParallelMultiheadAttention,
)

from spmd.tensor.parallel.api import (
    tp_shard_self_attn,
    replicate_input,
    replicate_output,
)

from spmd.tensor.parallel.style import (
<<<<<<< HEAD
=======
    ParallelStyle,
    make_input_shard_1d,
    make_input_replicate_1d,
>>>>>>> main
    make_output_shard_1d,
    make_output_replicate_1d,
    make_output_tensor,
)
