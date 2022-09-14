# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import DTensor
from spmd.tensor.ops.common_rules import reduction_rule


reduction_ops = [
    "aten.all.default",
    "aten.sum.SymInt",
    "aten.sum.default",
    "aten.sum.dim_IntList",
]

for reduction_op in reduction_ops:
    DTensor._op_to_rules[reduction_op] = reduction_rule
