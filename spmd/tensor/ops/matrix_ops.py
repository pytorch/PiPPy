# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import Tuple, Dict
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.common_rules import einop_rule, pointwise_rule
from spmd.tensor.ops.utils import register_prop_rule


def _update_schema_suggestion_for_addmm(
    output_sharding: OutputSharding,
    args_schema: Tuple[object, ...],
    kwargs_schema: Dict[str, object],
    pointwise_add_update: bool = True,
) -> OutputSharding:
    # schema suggestion coming from output sharding could be:
    # 1. pointwise add sharding input suggestion
    # 2. mm sharding input suggestion
    # inplace update schema suggestion to return addmm suggestion
    assert output_sharding.schema_suggestions is not None
    suggestion = output_sharding.schema_suggestions[0]
    if pointwise_add_update:
        # update with pointwise suggestion
        args_schema = (
            suggestion.args_schema[0],
            args_schema[1],
            args_schema[2],
        )
    else:
        # update with mm suggestion
        args_schema = (
            args_schema[0],
            suggestion.args_schema[0],
            suggestion.args_schema[1],
        )

    output_sharding.schema_suggestions = [
        OpSchema(
            args_schema=args_schema,
            kwargs_schema=kwargs_schema,
        )
    ]
    return output_sharding


@register_prop_rule("aten.mm.default")
def mm_rules(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("mk,kn->mn", op_schema, linearity=False)


@register_prop_rule("aten.addmm.default")
def addmm_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec, mat1_spec, mat2_spec = op_schema.args_spec
    mm_out_sharding = mm_rules(OpSchema((mat1_spec, mat2_spec), {}))
    if mm_out_sharding.output_spec is None:
        # non-eligible input, suggest addmm input specs
        if mm_out_sharding.schema_suggestions is not None:
            # TODO: add more suggestions for resharding
            return _update_schema_suggestion_for_addmm(
                mm_out_sharding,
                op_schema.args_schema,
                op_schema.kwargs_schema,
                pointwise_add_update=False,
            )
        else:
            return OutputSharding(None)

    # run point wise rule on input + (mm_out) with linearity
    output_sharding = pointwise_rule(
        OpSchema((input_spec, mm_out_sharding.output_spec), {}), linearity=True
    )
    # if propagation failed, edit the schema suggestion from pointwise rules
    # to return addmm suggestion instead as it's a chained suggestion.
    if (
        output_sharding.output_spec is None
        and output_sharding.schema_suggestions is not None
    ):
        return _update_schema_suggestion_for_addmm(
            output_sharding, op_schema.args_schema, op_schema.kwargs_schema
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
    bmm_output_sharding = bmm_rules(OpSchema((mat1_spec, mat2_spec), {}))
    if bmm_output_sharding.output_spec is None:
        # TODO: add more suggestions
        if bmm_output_sharding.schema_suggestions is not None:
            return _update_schema_suggestion_for_addmm(
                bmm_output_sharding,
                op_schema.args_schema,
                op_schema.kwargs_schema,
                pointwise_add_update=False,
            )
        else:
            return OutputSharding(None)

    # run point wise rule on input + (bmm_out) with linearity
    output_sharding = pointwise_rule(
        OpSchema((input_spec, bmm_output_sharding.output_spec), {}),
        linearity=True,
    )
    # if propagation failed, edit the schema suggestion from pointwise rules
    # to return baddbmm suggestion instead as it's a chained suggestion.
    if (
        output_sharding.output_spec is None
        and output_sharding.schema_suggestions is not None
    ):
        return _update_schema_suggestion_for_addmm(
            output_sharding, op_schema.args_schema, op_schema.kwargs_schema
        )

    return output_sharding
