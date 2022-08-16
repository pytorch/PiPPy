# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import Optional
from spmd.tensor.dispatch import OpSchema
from spmd.tensor.placement_types import PlacementSpec
from spmd.tensor.ops.prop_rules import einop_prop, mm_prop, pointwise_prop
from spmd.tensor.ops.utils import register_prop_rule


@register_prop_rule("aten.mm.default")
def mm_rules(op_schema: OpSchema) -> Optional[PlacementSpec]:
    mat1_spec, mat2_spec = op_schema.args_spec
    return mm_prop(mat1_spec, mat2_spec)


@register_prop_rule("aten.addmm.default")
def addmm_rules(op_schema: OpSchema) -> Optional[PlacementSpec]:
    input_spec, mat1_spec, mat2_spec = op_schema.args_spec
    mm_out_spec = mm_prop(mat1_spec, mat2_spec)
    if mm_out_spec is None:
        return None
    return pointwise_prop((input_spec, mm_out_spec))


@register_prop_rule("aten.t.default")
def transpose_rule(op_schema: OpSchema) -> Optional[PlacementSpec]:
    return einop_prop("ij->ji", op_schema.args_spec)
