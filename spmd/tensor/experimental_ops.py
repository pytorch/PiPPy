# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, Optional, Sequence

import torch

from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
    _Partial,
)
from torch.distributed._tensor.dispatch import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.ops.common_rules import pointwise_rule


@register_prop_rule("aten.native_layer_norm.default")
def _prop_native_layer_norm(op_schema: OpSchema) -> OutputSharding:
    input, normalized_shape, weight, bias, eps = op_schema.args_schema
    assert isinstance(input, DTensorSpec)
    assert isinstance(weight, DTensorSpec)
    assert isinstance(bias, DTensorSpec)
    assert isinstance(normalized_shape, (tuple, list))
    assert all(isinstance(p, Replicate) for p in weight.placements)
    assert all(isinstance(p, Replicate) for p in bias.placements)
    # only the left-most (non-normalized) dimensions of the input can be sharded
    batch_ndim = len(input.shape) - len(normalized_shape)
    assert all(
        isinstance(p, Replicate)
        or (isinstance(p, Shard) and p.dim < batch_ndim,)
        for p in input.placements
    )
    stats_spec = DTensorSpec(
        mesh=weight.mesh,
        placements=input.placements,
        shape=torch.Size(
            input.shape[:batch_ndim] + (1,) * len(normalized_shape)
        ),
        ndim=input.ndim,
    )
    return OutputSharding(output_spec=(input, stats_spec, stats_spec))


@register_prop_rule("aten.native_layer_norm_backward.default")
def _prop_native_layer_norm_backward(op_schema: OpSchema) -> OutputSharding:
    (
        grad,
        input,
        normalized_shape,
        result1,
        result2,
        weight,
        bias,
        grad_input_mask,
    ) = op_schema.args_schema
    assert isinstance(grad, DTensorSpec)
    assert isinstance(weight, DTensorSpec)
    assert isinstance(bias, DTensorSpec)
    assert isinstance(grad_input_mask, (list, tuple))
    assert all(isinstance(s, Replicate) for s in weight.placements)
    assert all(isinstance(s, Replicate) for s in bias.placements)
    # ensure sharding on dim 0, which will trigger the "Partial" output on weight and bias grads
    assert any(
        isinstance(s, Shard) and s.dim == 0 for s in grad.placements
    ), f"Got {grad.placements}"
    weight_grad = DTensorSpec(
        mesh=weight.mesh,
        placements=[_Partial()] * weight.mesh.ndim,
        shape=weight.shape,
        ndim=weight.ndim,
    )
    bias_grad = DTensorSpec(
        mesh=bias.mesh,
        placements=[_Partial()] * bias.mesh.ndim,
        shape=bias.shape,
        ndim=bias.ndim,
    )
    return OutputSharding(
        # NOTE: type errors below are legit. This is because DTensor currently
        # doesn't support Optional return values. Need to be fixed in DTensor repo.
        output_spec=(
            # type: ignore
            grad if grad_input_mask[0] else None,
            # type: ignore
            weight_grad if grad_input_mask[1] else None,
            # type: ignore
            bias_grad if grad_input_mask[2] else None,
        ),
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
            output_placements = tensor.placements  # type: ignore
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
) -> Sequence[Placement]:
    """
    Considers 2 first inputs of op_schema as having same shape,
    and returns suggested placement for a pointwise operation.
    """
    # consider the operating dimension as a singleton to prevent sharding on it
    # however, if active_dim is None, this means the input and output shapes are equal and
    # we'll apply exactly the pointwise rule.
    args_schema = [
        DTensorSpec(
            mesh=s.mesh,  # type: ignore
            placements=s.placements,  # type: ignore
            shape=s.shape[0:active_dim] + (1,) + s.shape[active_dim + 1 :]  # type: ignore
            if active_dim is not None
            else s.shape,  # type: ignore
        )
        for s in op_schema.args_schema[:2]
    ]

    op_schema = OpSchema(
        func_schema=op_schema.func_schema,
        args_schema=args_schema,  # type: ignore
        kwargs_schema={},
        is_inplace=op_schema.is_inplace,
        is_out_variant=op_schema.is_out_variant,
    )
    output_sharding = pointwise_rule(op_schema, linearity=False)  # type: ignore
    if output_sharding.output_spec:
        assert isinstance(output_sharding.output_spec, DTensorSpec)
        return output_sharding.output_spec.placements  # type: ignore
    else:
        assert output_sharding.schema_suggestions is not None
        out_schema = output_sharding.schema_suggestions[0].args_schema[0]
        assert isinstance(out_schema, DTensorSpec)
        return tuple(out_schema.placements)


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
        assert end >= src.shape[dim]  # type: ignore
        dim = None

    # apply sharding refinement as implemented in pointwise_rule
    input_suggestion = list(_refine_sharding(op_schema, dim))
    # apply the exception -- disallow sharding on the operating dimension.
    for i, p in enumerate(input_suggestion):
        if isinstance(p, Shard) and p.dim == dim:
            input_suggestion[i] = Replicate()
    input_suggestion = tuple(input_suggestion)  # type: ignore

    if input_suggestion == tuple(input.placements) and src.placements == tuple(
        input.placements
    ):
        # if our sharding is correct, the output sharding will be the same as the input.
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=input.mesh,
                placements=input.placements,
                shape=input.shape,
                ndim=input.ndim,
            )
        )
    else:
        # otherwise, return the suggestion.
        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(
                        DTensorSpec(
                            mesh=input.mesh,
                            placements=input_suggestion,
                            shape=input.shape,
                            ndim=input.ndim,
                        ),
                        DTensorSpec(
                            mesh=src.mesh,
                            placements=input_suggestion,
                            shape=src.shape,
                            ndim=src.ndim,
                        ),
                    )
                    + op_schema.args_schema[2:],
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
        )
