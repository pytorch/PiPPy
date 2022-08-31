# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.math_ops import einop_rule
from spmd.tensor.ops.pointwise_ops import pointwise_rule
from spmd.tensor.ops.utils import register_prop_rule


@register_prop_rule("aten.mm.default")
def mm_rules(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("mk,kn->mn", op_schema, linearity=False)


@register_prop_rule("aten.addmm.default")
def addmm_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec, mat1_spec, mat2_spec = op_schema.args_spec
    mm_out_spec = mm_rules(OpSchema((mat1_spec, mat2_spec), {})).output_spec
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

@register_prop_rule("aten.bmm.default")
def bmm_rules(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("bmk,bkn->bmn", op_schema, linearity=False)

@register_prop_rule("aten.baddbmm.default")
def baddbmm_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec, mat1_spec, mat2_spec = op_schema.args_spec
    mm_out_spec = bmm_rules(OpSchema((mat1_spec, mat2_spec), {})).output_spec
    if (bmm_output_spec is None):
        # TODO: add suggestion
        return OutputSharding(None)

    # run point wise rule on input + (bmm_out) with linearity
    output_sharding = pointwise_rule(
        OpSchema((input_spec, bmm_output_spec), {}), linearity=True
    )
    # if propagation failed, edit the schema suggestion from pointwise rules
    # to return baddbmm suggestion instead as it's a chained suggestion.
    if (
        output_sharding.output_spec is None
        and output_sharding.schema_suggestions is not None
    ):
        pointwise_suggestion = output_sharding.schema_suggestions[0]
        output_sharding.schema_suggestions[0] = OpSchema(
            args_schema=(
                pointwise_suggestion.args_schema[0],
                mat1_spec,
                mat2_spec,
            ),
            kwargs_schema=op_schema.kwargs_schema,
        )

    return output_sharding
