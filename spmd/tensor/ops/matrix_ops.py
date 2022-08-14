# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.prop_rules import einop_prop, mm_prop, pointwise_prop
from spmd.tensor.placement_types import _Partial, PlacementSpec, Shard
from spmd.tensor.ops.utils import register_prop_rule


@register_prop_rule("aten.mm.default")
def mm_rules(op_schema: OpSchema) -> OutputSharding:
    mat1_spec, mat2_spec = op_schema.args_spec
    return OutputSharding(mm_prop(mat1_spec, mat2_spec))


@register_prop_rule("aten.addmm.default")
def addmm_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec, mat1_spec, mat2_spec = op_schema.args_spec
    mm_out_spec = mm_prop(mat1_spec, mat2_spec)
    if mm_out_spec is None:
        # non-eligible input, suggest addmm input specs
        # TODO: add suggested input specs for resharding
        return OutputSharding(None)
    # TODO: add multi dim support for addmm
    if (
        mm_out_spec.placements[0].is_partial()
        and input_spec.placements[0].is_replicate()
    ):
        # for the case where mm out is partial and input is replicate
        # we should suggest the input be a partial instead of replicate
        suggested_input_spec = PlacementSpec(
            input_spec.ndim, input_spec.mesh, [_Partial()]
        )
        suggested_mat1_spec = PlacementSpec(
            mat1_spec.ndim, mat1_spec.mesh, [Shard(1)]
        )
        suggested_mat2_spec = PlacementSpec(
            mat2_spec.ndim, mat2_spec.mesh, [Shard(0)]
        )
        return OutputSharding(
            None,
            schema_suggestions=[
                OpSchema(
                    (
                        suggested_input_spec,
                        suggested_mat1_spec,
                        suggested_mat2_spec,
                    ),
                    op_schema.kwargs_schema,
                )
            ],
        )

    return OutputSharding(
        pointwise_prop((input_spec, mm_out_spec), linearity=True)
    )


@register_prop_rule("aten.t.default")
def transpose_rule(op_schema: OpSchema) -> OutputSharding:
    return OutputSharding(einop_prop("ij->ji", op_schema.args_spec))
