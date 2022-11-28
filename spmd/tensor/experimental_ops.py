# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed._tensor.dispatch import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule


@register_prop_rule("aten.native_layer_norm.default")
def _prop_native_layer_norm(op_schema: OpSchema) -> OutputSharding:
    # TODO: Assert placement is Replicate() since we need to insert
    # allreduce calls for _Partial results.
    input, normalized_shape, weight, bias, eps = op_schema.args_schema
    out, mean, rstd = input, weight, weight
    dims_list = op_schema.args_schema[1]
    assert isinstance(input, DTensorSpec)
    return OutputSharding(
        output_spec=(
            DTensorSpec(
                mesh=input.mesh,
                placements=tuple(input.placements),
                ndim=input.ndim,
                shape=torch.Size(s for (i, s) in enumerate(input.shape)),
            ),
            DTensorSpec(
                mesh=input.mesh,
                placements=tuple(input.placements),
                ndim=input.ndim,
                shape=torch.Size(s for (i, s) in enumerate(input.shape)),
            ),
            DTensorSpec(
                mesh=input.mesh,
                placements=tuple(input.placements),
                ndim=input.ndim,
                shape=torch.Size(s for (i, s) in enumerate(input.shape)),
            ),
        )
    )
