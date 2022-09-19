# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import DTensor, DTensorSpec, Replicate, _Partial
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.utils import register_prop_rule


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


default_prop_ops = [
    "aten._to_copy.default",
    "aten.clone.default",
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

no_shard_prop_ops = ["aten._local_scalar_dense.default"]

for op in default_prop_ops:
    DTensor._op_to_rules[op] = default_prop_rule

for op in create_like_ops:
    DTensor._op_to_rules[op] = prop_create_like

for op in no_shard_prop_ops:
    DTensor._op_to_rules[op] = no_shard_prop_rule


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
