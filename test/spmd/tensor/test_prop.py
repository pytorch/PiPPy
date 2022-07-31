# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from spmd.tensor.ops.math_ops import einop_prop
from spmd.tensor.placement_types import Shard, Replicate, _Partial


if __name__ == "__main__":
    run_tests()
