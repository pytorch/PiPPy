# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast, List, Optional, Sequence, Tuple

import torch

from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed._tensor.api import DTensor
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


@register_prop_rule("aten.cat.default")		
def prop_cat(op_schema: OpSchema) -> OutputSharding:
    tensor_list = op_schema.args_schema[0]
    if len(op_schema.args_schema) > 1:
        dim = op_schema.args_schema[1]
    else:
        dim = -1
    assert isinstance(tensor_list, (list, tuple))
    assert isinstance(dim, int)

    if dim < 0:
        dim += len(tensor_list[0].shape)

    output_placements: Optional[List[Placement]] = None
    for tensor in tensor_list:
        assert isinstance(tensor, DTensorSpec)
        if output_placements is None:
            output_placements = tensor.placements
        else:
            # not sure why we're getting a tuple here sometimes, so casting to list to be sure
            assert list(output_placements) == list(
                tensor.placements
            ), f"Current only accept cat when all inputs are same sharded. Got {output_placements} vs. {tensor.placements}"
    assert output_placements is not None

    output_shape = list(tensor_list[0].shape)
    output_shape[dim] = sum((t.shape[dim] for t in tensor_list))

    return OutputSharding(
        output_spec=DTensorSpec(
            mesh=tensor_list[0].mesh,
            placements=output_placements,
            ndim=tensor_list[0].ndim,
            shape=torch.Size(output_shape),
        )
    )


def _refine_sharding(
    op_schema: OpSchema, active_dim: Optional[int]
) -> Tuple[Placement]:
    """
    Considers 2 first inputs of op_schema as having same shape,
    and returns suggested placement for a pointwise operation.
    """
    # consider the operating dimension as a singleton to prevent sharding on it
    # however, if active_dim is None, this means the input and output shapes are equal and
    # we'll apply exactly the pointwise rule.
    args_schema = [
        DTensorSpec(
            mesh=s.mesh,
            placements=s.placements,
            shape=s.shape[0:active_dim] + (1,) + s.shape[active_dim + 1 :]
            if active_dim is not None
            else s.shape,
        )
        for s in op_schema.args_schema[:2]
    ]

    op_schema = OpSchema(
        func_schema=op_schema.func_schema,
        args_schema=args_schema,
        kwargs_schema={},
        is_inplace=op_schema.is_inplace,
        is_out_variant=op_schema.is_out_variant,
    )
    output_sharding = pointwise_rule(op_schema, linearity=False)
    if output_sharding.output_spec:
        assert isinstance(output_sharding.output_spec, DTensorSpec)
        return output_sharding.output_spec.placements
    else:
        return output_sharding.schema_suggestions[0].args_schema[0].placements


@register_prop_rule("aten.slice_scatter.default")
def prop_slice_scatter(op_schema: OpSchema) -> OutputSharding:
    # 1. number of dimensions in input and src need to match.
    # 2. number of elements on all non-dim need to match between input and src.
    # 3. numer of elements in src in dim need to match the slice size.
    # Given the above:
    # - We suggest for src to follow the sharding of input, except on the scatter dimension,
    #   where our best bet for now is to make them replicated as a fall-back.
    #   TODO: Ideally we'd like to make sure the output is re-sharded afterwards to keep input sharding.

    defaults = (None, None, 0, None, None, 1)
    input, src, dim, start, end, step = (
        op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    )
    assert isinstance(input, DTensorSpec)
    assert isinstance(src, DTensorSpec)
    assert isinstance(dim, int)

    if dim < 0:
        dim += input.ndim

    # if the input shape and the output shape are the same on the operating dimension,
    # this is effectively a no-op, so we just propagate sharding as we would do for
    # pointwise, no exceptions.
    if input.shape[dim] == src.shape[dim]:
        assert start == 0
        assert end >= src.shape[dim]
        dim = None

    # apply sharding refinement as implemented in pointwise_rule
    input_suggestion = list(_refine_sharding(op_schema, dim))
    # apply the exception -- disallow sharding on the operating dimension.
    for i, p in enumerate(input_suggestion):
        if isinstance(p, Shard) and p.dim == dim:
            input_suggestion[i] = Replicate()
    input_suggestion = tuple(input_suggestion)
