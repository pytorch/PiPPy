# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import Optional
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.math_ops import einop_rule
from spmd.tensor.ops.pointwise_ops import pointwise_rule
from spmd.tensor.placement_types import _Partial, DTensorSpec
from spmd.tensor.ops.utils import register_prop_rule


def mm_prop(
    mat1_spec: DTensorSpec, mat2_spec: DTensorSpec
) -> Optional[DTensorSpec]:
    # mm propagation rule:
    # mat1: shard(0),  mat2: replicate
    # mat1: replicate, mat2: shard(1)
    # mat1: shard(1),  mat2: shard(0)
    # propagation rules only propagates the combs without communication
    # TODO: support multi-dim device mesh op with einop propagation
    if (
        mat1_spec.placements[0].is_shard(dim=0)
        and mat2_spec.placements[0].is_replicate()
    ):
        return mat1_spec
    elif mat1_spec.placements[0].is_replicate() and mat2_spec.placements[
        0
    ].is_shard(dim=1):
        return mat2_spec
    elif mat1_spec.placements[0].is_shard(dim=1) and mat2_spec.placements[
        0
    ].is_shard(dim=0):
        placements = [_Partial()]
        return DTensorSpec(mat1_spec.mesh, placements, shape=mat1_spec.shape)
    elif (
        mat1_spec.placements[0].is_replicate()
        and mat2_spec.placements[0].is_replicate()
    ):
        return mat1_spec
    else:
        # TODO(anj): Add logging (or something else) to indicate no output sharding
        # spec specified. If we don't have autoboxing rules specified in the op below,
        # we will throw an error without a populated `failed_reason`.
        # not local compute, need to rely on auto redistribute, return None
        return None


def mm_rules(op_schema: OpSchema) -> OutputSharding:
    mat1_spec, mat2_spec = op_schema.args_spec
    return OutputSharding(mm_prop(mat1_spec, mat2_spec))


default_mm_ops = ["aten.mm.default", "aten.mul.Tensor"]

for mm_op in default_mm_ops:
    DTensor._op_to_rules[mm_op] = mm_rules


@register_prop_rule("aten.addmm.default")
def addmm_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec, mat1_spec, mat2_spec = op_schema.args_spec
    mm_out_spec = mm_prop(mat1_spec, mat2_spec)
    if mm_out_spec is None:
        # non-eligible input, suggest addmm input specs
        # TODO: add suggested input specs for resharding
        return OutputSharding(None)
    # TODO: add multi dim support for addmm

    # run point wise rule on input + (mm_out) with linearity
    output_sharding = pointwise_rule(
        OpSchema((input_spec, mm_out_spec), {}), linearity=True
    )
    # if propagation failed, edit the schema suggestion from pointwise rules
    # to return addmm suggestion instead as it's a chained suggestion.
    if (
        output_sharding.output_spec is None
        and output_sharding.schema_suggestions is not None
    ):
        pointwise_suggestion = output_sharding.schema_suggestions[0]
        assert output_sharding.schema_suggestions is not None
        output_sharding.schema_suggestions[0] = OpSchema(
            args_schema=(
                pointwise_suggestion.args_schema[0],
                mat1_spec,
                mat2_spec,
            ),
            kwargs_schema=op_schema.kwargs_schema,
        )

    return output_sharding


@register_prop_rule("aten.t.default")
def transpose_rule(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("ij->ji", op_schema)
