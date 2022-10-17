# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from spmd.tensor.api import (
    DTensor,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
    _Partial,
)
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.common_rules import pointwise_rule
from spmd.tensor.ops.utils import register_prop_rule
from typing import List, Optional, Sequence, Tuple, cast


# NOTE: the default propagation rule should apply for
# any operator that does not return a DTensor, i.e.
# for operators that only returns int/float/bool, we by
# default still propagate the spec, this is to ensure
# that we only return None for the case where the sharding
# propagation failed, and we should do auto-redistribute
def default_prop_rule(op_schema: OpSchema) -> OutputSharding:
    # by default prop the first arg spec
    return OutputSharding(op_schema.args_spec[0])


def prop_create_like(op_schema: OpSchema) -> OutputSharding:
    # For operators that create tensors with same shape as input but
    # with specific content that does not depend on the input, we
    # can propagate Sharding, but we have to make sure we move from
    # partial to replicated.
    input_spec = op_schema.args_spec[0]
    output_spec = DTensorSpec(
        mesh=input_spec.mesh,
        placements=tuple(
            Replicate() if isinstance(p, _Partial) else p
            for p in input_spec.placements
        ),
        ndim=input_spec.ndim,
        shape=input_spec.shape,
    )
    return OutputSharding(output_spec=output_spec)


# some tensor ops should not support shard, i.e. local_scalar_dense
# shouldn't work for shard as it requires numel == 1
def no_shard_prop_rule(op_schema: OpSchema) -> OutputSharding:
    # by default prop the first arg spec
    tensor_spec = op_schema.args_spec[0]
    for placement in tensor_spec.placements:
        if placement.is_shard():
            return OutputSharding(
                None,
                failed_reason=f"Op does not support input placements "
                f"with `Shard`, but found placements: "
                f"{tensor_spec.placements}",
            )
    # otherwise default prop the first arg spec
    return OutputSharding(tensor_spec)


def new_factory_rule(op_schema: OpSchema) -> OutputSharding:
    # this op would benefit from backward sharding propagation!
    # Since we cannot do that yet, just return replicated
    input = op_schema.args_schema[0]
    size = torch.Size(cast(Sequence[int], op_schema.args_schema[1]))
    assert isinstance(input, DTensorSpec)

    return OutputSharding(
        output_spec=DTensorSpec(
            mesh=input.mesh,
            placements=[Replicate()] * input.mesh.ndim,
            shape=size,
            ndim=len(size),
        )
    )


default_prop_ops = [
    "aten._to_copy.default",
    "aten.clone.default",
    "aten.contiguous.default",
    "aten.copy_.default",
    "aten.detach.default",
    "aten.is_same_size.default",
    "aten.new_empty_strided.default",
]

create_like_ops = [
    "aten.empty_like.default",
    "aten.fill_.Scalar",
    "aten.full_like.default",
    "aten.ones_like.default",
    "aten.zero_.default",
    "aten.zeros_like.default",
]

new_factory_ops = [
    "aten.new_full.default",
    "aten.new_ones.default",
    "aten.new_zeros.default",
]

no_shard_prop_ops = ["aten._local_scalar_dense.default"]

for op in default_prop_ops:
    DTensor._op_to_rules[op] = default_prop_rule

for op in create_like_ops:
    DTensor._op_to_rules[op] = prop_create_like

for op in no_shard_prop_ops:
    DTensor._op_to_rules[op] = no_shard_prop_rule

for op in new_factory_ops:
    DTensor._op_to_rules[op] = new_factory_rule


@register_prop_rule("aten.bucketize.Tensor")
def prop_bucketize(op_schema: OpSchema) -> OutputSharding:
    """
    Point-wise on the first input (just propagate input sharding).
    Expect replicated for second input.
    """
    input_schema, boundaries = op_schema.args_schema
    assert isinstance(input_schema, DTensorSpec)
    assert isinstance(boundaries, DTensorSpec)

    if all(isinstance(p, Replicate) for p in boundaries.placements):
        return OutputSharding(output_spec=input_schema)
    else:
        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(
                        input_schema,
                        DTensorSpec(
                            mesh=boundaries.mesh,
                            placements=[Replicate()]
                            * len(boundaries.placements),
                            ndim=boundaries.ndim,
                            shape=boundaries.shape,
                        ),
                    ),
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
        )


def unshard_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> Sequence[Placement]:
    """Disallow the given tensor dimension to be sharded"""
    return tuple(
        p if (not isinstance(p, Shard) or p.dim != dim) else Replicate()
        for p in placements
    )


def _prop_all_but_dim(
    op_schema: OpSchema, dim: int, out_shape: torch.Size
) -> OutputSharding:
    """
    Considering an op that takes its input as first argument, forwards all shardings
    except for the given dimension.
    """
    input_spec = op_schema.args_schema[0]
    assert isinstance(input_spec, DTensorSpec)

    output_placements = unshard_tensor_dim(input_spec.placements, dim=dim)
    output_spec = DTensorSpec(
        mesh=input_spec.mesh,
        placements=output_placements,
        shape=out_shape,
        ndim=input_spec.ndim,
    )

    if input_spec.placements == output_placements:
        out = OutputSharding(output_spec=output_spec)
    else:
        suggested_input_spec = DTensorSpec(
            mesh=input_spec.mesh,
            placements=output_placements,
            ndim=input_spec.ndim,
            shape=input_spec.shape,
        )
        out = OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(suggested_input_spec,)
                    + op_schema.args_schema[1:],
                    kwargs_schema=op_schema.kwargs_schema,
                ),
            ],
        )
    return out


@register_prop_rule("aten.slice.Tensor")
def prop_slice(op_schema: OpSchema) -> OutputSharding:
    """NOTE: can be further optimized (right now it replicates before slicing on a sharded dimension)"""
    defaults = (None, 0, None, None, 1)
    input_spec, dim, start, end, step = (
        op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    )
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(dim, int)
    assert start is None or isinstance(start, int)
    assert end is None or isinstance(end, int)
    assert isinstance(step, int)

    # normalize arguments
    if dim < 0:
        dim += input_spec.ndim
    if start is None:
        start = 0
    if step is None:
        step = 1
    if end is None or end > input_spec.shape[dim]:
        end = input_spec.shape[dim]
    if end < 0:
        end += input_spec.shape[dim]

    if start == 0 and end == input_spec.shape[dim] and step == 1:
        return OutputSharding(output_spec=input_spec)

    # shape propagation
    slice_len = (end - start + step - 1) // step
    out_shape = torch.Size(
        tuple(input_spec.shape[0:dim])
        + (slice_len,)
        + tuple(input_spec.shape[dim + 1 :])
    )

    return _prop_all_but_dim(op_schema, dim=dim, out_shape=out_shape)


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

    # first, we keep the input sharding, except for the input dimension
    # also, we cannot allow partial sum anymore.
    input_suggestion = tuple(
        Replicate()
        if isinstance(p, _Partial) or (isinstance(p, Shard) and p.dim == dim)
        else p
        for p in input.placements
    )

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


@register_prop_rule("aten.index_select.default")
def prop_index_select(op_schema: OpSchema) -> OutputSharding:
    values_spec, dim, indices_spec = op_schema.args_schema

    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(dim, int)
    assert isinstance(indices_spec, DTensorSpec)

    all_indices_spec: List[Optional[DTensorSpec]] = [
        indices_spec if dim == i else None for i in range(values_spec.ndim)
    ]

    result = prop_index(
        OpSchema(
            func_schema=op_schema.func_schema,
            args_schema=(values_spec, all_indices_spec),
            kwargs_schema=op_schema.kwargs_schema,
        )
    )
    if result.schema_suggestions:
        result.schema_suggestions = [
            OpSchema(
                func_schema=op_schema.func_schema,
                args_schema=(s.args_schema[0], dim, s.args_schema[1][dim]),
                kwargs_schema=op_schema.kwargs_schema,
            )
            for s in result.schema_suggestions
        ]
    return result


@register_prop_rule("aten.index.Tensor")
def prop_index(op_schema: OpSchema) -> OutputSharding:
    """
    Expect replicated on the first input; _mostly_ pointwise on the second input.
    TODO: exception: when the dtype of second input is "bool", then a torch.nonzero needs to be triggered first.
    """
    # Current sharding constraints:
    # For values:
    #   1. We currently require that the dimension of values_spec be replicated or partial
    #      if they are being indexed on.
    #   2. Other dimensions of values_spec can remain sharded if they are so.
    # For indices:
    #   Indices can be either sharded or replicated. All index tensors need to be sharded
    #   in a compatible way, following the pointwise rule (including resolving _Partial
    #   into either sharded or replicated)

    values_spec, multi_indices_spec = op_schema.args_schema
    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(multi_indices_spec, list)
    multi_indices_spec = cast(List[Optional[DTensorSpec]], multi_indices_spec)
    valid_indices_spec: List[Tuple[int, DTensorSpec]] = [
        (i, a) for i, a in enumerate(multi_indices_spec) if a is not None
    ]

    # 1. All indices have to be sharded equally. Moreover, indices can be broadcast.
    #    Here, we piggyback on the pointwise sharding rule for indices.
    indices_out = pointwise_rule(
        OpSchema(
            func_schema=op_schema.func_schema,
            args_schema=tuple(v[1] for v in valid_indices_spec),
            kwargs_schema={},
        )
    )
    need_reshard_on_indices = indices_out.output_spec is None

    if not need_reshard_on_indices:
        # this means that our inputs are already sharded properly and we will use that as our indices_spec
        assert isinstance(indices_out.output_spec, DTensorSpec)
        indices_spec: DTensorSpec = indices_out.output_spec
    else:
        assert indices_out.schema_suggestions is not None
        valid_indices_suggestion = indices_out.schema_suggestions[0]
        for i, v in enumerate(valid_indices_suggestion.args_spec):
            multi_indices_spec[valid_indices_spec[i][0]] = v
        # we'll need to call pointwise_rule again to see what's our ideal indices_spec and then
        # use that to compute our ideal values_spec
        indices_output_spec = pointwise_rule(
            valid_indices_suggestion
        ).output_spec
        assert isinstance(indices_output_spec, DTensorSpec)
        indices_spec = indices_output_spec

    lookup_dims = set(v[0] for v in valid_indices_spec)

    need_reshard_on_values = tuple(
        (
            isinstance(vp, Shard)
            and (vp.dim in lookup_dims or isinstance(ip, Shard))
        )
        for vp, ip in zip(values_spec.placements, indices_spec.placements)
    )

    if not need_reshard_on_indices and not any(need_reshard_on_values):

        value_placements = values_spec.placements
        value_shape = values_spec.shape

        all_dims_consecutive = all(
            b[0] - a[0] == 1
            for b, a in zip(valid_indices_spec[1:], valid_indices_spec[:-1])
        )
        if all_dims_consecutive:
            # if all index vectors are consecutives, insert at the dimension of the first index
            insert_dim: int = valid_indices_spec[0][0]
        else:
            # else, insert on the first dimension
            insert_dim = 0

        def place(vp: Placement, ip: Placement) -> Placement:
            if isinstance(vp, Shard):
                return Shard(
                    vp.dim
                    if vp.dim < insert_dim
                    # accounts for the offset in output dimensions
                    else vp.dim
                    + indices_spec.ndim
                    - sum(1 if vp.dim > v[0] else 0 for v in valid_indices_spec)
                )
            if isinstance(ip, Shard):
                return Shard(ip.dim + insert_dim)
            # _Partial or Replicated
            return vp

        value_placements = tuple(
            place(vp, ip)
            for vp, ip in zip(values_spec.placements, indices_spec.placements)
        )
        value_shape = torch.Size(
            tuple(value_shape[:insert_dim])
            + tuple(indices_spec.shape)
            + tuple(value_shape[insert_dim + 1 :])
        )

        result = OutputSharding(
            output_spec=DTensorSpec(
                mesh=values_spec.mesh,
                placements=value_placements,
                shape=value_shape,
                ndim=len(value_shape),
            )
        )
        return result
    else:
        result = OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(
                        DTensorSpec(
                            mesh=values_spec.mesh,
                            placements=[
                                Replicate() if need_reshard_on_values[i] else v
                                for i, v in enumerate(values_spec.placements)
                            ],
                            ndim=values_spec.ndim,
                            shape=values_spec.shape,
                        ),
                        multi_indices_spec,
                    ),
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
        )
        return result
