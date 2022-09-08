# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, Dict, Tuple, cast
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.placement_types import _Partial, DTensorSpec
from spmd.tensor.ops.utils import as_list, register_prop_rule


def _gen_spec_with_pending_sum(
    spec: DTensorSpec, pending_sum: List[int]
) -> DTensorSpec:
    new_placements = [
        placement if i not in pending_sum else _Partial()
        for i, placement in enumerate(spec.placements)
    ]
    return DTensorSpec(
        spec.mesh, new_placements, shape=spec.shape, ndim=spec.ndim
    )


def _gen_reshard_suggestions(
    input_dims: List[str],
    input_specs: Tuple[DTensorSpec, ...],
    dim_to_sharding: Dict[str, int],
    pending_sum: List[int],
) -> OutputSharding:
    suggested_arg_specs: List[DTensorSpec] = []
    for input_dim, input_spec in zip(input_dims, input_specs):
        dim_map = [dim_to_sharding[dim] for dim in input_dim]
        suggested_arg_specs.append(
            DTensorSpec.from_dim_map(
                mesh=input_spec.mesh, dim_map=dim_map, sums=pending_sum
            )
        )
    return OutputSharding(
        None,
        schema_suggestions=[OpSchema(tuple(suggested_arg_specs), {})],
        failed_reason="Input placements op sharding propagation failed, need to reshard!",
    )


def einop_rule(
    equation: str, op_schema: OpSchema, linearity: bool = False
) -> OutputSharding:
    """
    Propagate the sharding of inputs to output for ops whose data
    moves according to einsum notation. This is mostly borrowed
    from @zdevito's sharding simulator. Examples:
        mk,kn->mn - einsum
        ij,ij->ij - addition
        ij,j->ij - broadcasted addition
        ij->i - reduction
    Other ops could use this propagation algorithm when applied, note
    that einsum propagation only deal with list of specs (DTensor specs)
    as it only works on list of tensors!
    """
    # parse einop equation and extract arg specs
    inputs, outputs = equation.split("->")
    input_dims, output_dims = inputs.split(","), outputs.split(",")
    input_specs = op_schema.args_spec
    # NOTE: only support single output unless needed in future
    output_dim = output_dims[0]

    dim_to_sharding: Dict[str, int] = {}
    pending_sums: List[int] = []
    seen_shardings = {}
    needs_reshard = False
    # partial_linearity means if the op support linearity, and there exist
    # partial placements, all placements on the mesh dim should be partial
    partial_linearity = True

    # deal with partial placements, throw if it's not linearity op
    for mesh_dim in range(input_specs[0].mesh.ndim):
        for idx, spec in enumerate(input_specs):
            if spec.placements[mesh_dim].is_partial():
                if not linearity:
                    raise RuntimeError(
                        "Cannot do generic op on a tensor with partial sums"
                    )
                elif mesh_dim not in pending_sums and idx != 0:
                    # If the first input mesh dim is not partial
                    # then it does not conform with partial linearity
                    # property (all input mesh dim should be partial)
                    partial_linearity = False
                elif idx == 0:
                    seen_shardings[mesh_dim] = "+"
                # update pending sum list
                pending_sums.append(mesh_dim)
            else:
                if mesh_dim in pending_sums:
                    # fail if there's already pending sum on mesh dim!
                    partial_linearity = False

    if linearity and not partial_linearity:
        # it's a op that support linearity, but failed on partial linearity check
        # we fail the sharding propagation with suggestion to make all inputs
        # be partial on the corresponding mesh dim
        new_arg_specs = tuple(
            _gen_spec_with_pending_sum(spec, pending_sums)
            for spec in input_specs
        )
        return OutputSharding(
            None,
            schema_suggestions=[OpSchema(new_arg_specs, {})],
            failed_reason="Input placements does not satisfy linearity property of the op!",
        )

    def merge_sharding(dim: str, a: int, b: int) -> int:
        # merge the sharding of inputs if it's able to merge, i.e. we can merge
        # replicate and shard to shard, but this will trigger an reshard operation
        if a != b:
            if a == -1 or b == -1:
                # rehsard the replicate to match the sharded one
                nonlocal needs_reshard
                needs_reshard = True
                return a if a != -1 else b
            else:
                # TODO: further merge the sharding properly (i.e. reshard one input to replicate)
                raise RuntimeError(
                    f"{equation}: dim {dim} sharded two different ways: {a} and {b}"
                )
        else:
            return a

    for input_dim, input_spec in zip(input_dims, input_specs):
        for dim, mesh_dim in zip(input_dim, input_spec.dim_map):
            if (
                mesh_dim in seen_shardings
                and mesh_dim != -1
                and dim != seen_shardings[mesh_dim]
            ):
                # TODO: add recommendation to this case, i.e. all_gather one input
                raise RuntimeError(
                    "Two different input dims are sharded across the same mesh dim!"
                )

            seen_shardings[mesh_dim] = dim
            if dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
            else:
                dim_to_sharding[dim] = merge_sharding(
                    dim, dim_to_sharding[dim], mesh_dim
                )

    if needs_reshard:
        # TODO: merge this logic with partial linearity checks
        return _gen_reshard_suggestions(
            input_dims, input_specs, dim_to_sharding, pending_sums
        )

    # if no need to reshard, we directly generate the output sharding
    for dim, shard_on_mesh in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)

    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_specs[0].mesh,
            [dim_to_sharding[dim] for dim in output_dim],
            pending_sums,
        )
    )


def reduction_rule(op_schema: OpSchema) -> OutputSharding:
    """
    Propagate the sharding for reduction operations. Examples:
        ij->i - sum on dim
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # reduction op usually begin with a single tensor
    input_spec = cast(DTensorSpec, op_schema.args_schema[0])
    input_chars = alphabet[: input_spec.ndim]

    if len(op_schema.args_schema) > 1 and isinstance(
        op_schema.args_schema[1], (int, list)
    ):
        # this is usually a dim-based reduction op schema pattern, it
        # might not be true for every op for other special cases, we
        # need to specialize them as needed.
        # TODO: add support for things like `torch.unique` where it
        # does not follow the reduction op convention.
        dim_list = as_list(op_schema.args_schema[1])
        out_dimchars = input_chars.translate(
            {ord(alphabet[cast(int, dim)]): None for dim in dim_list}
        )
    else:
        # reducing to a single scalar tensor, we just mark output as empty
        out_dimchars = ""

    fmt = f"{input_chars}->{out_dimchars}"
    return einop_rule(fmt, op_schema)


reduction_ops = [
    "aten.all.default",
    "aten.sum.SymInt",
    "aten.sum.default",
    "aten.sum.dim_IntList",
]

for reduction_op in reduction_ops:
    DTensor._op_to_rules[reduction_op] = reduction_rule

@register_prop_rule("aten._softmax.default")
def softmax_rule(op_schema: OpSchema) -> OutputSharding:
    #print(type(op_schema.args_schema[0]))
    dim_map = op_schema.args_schema[0].dim_map
    softmax_dim = op_schema.args_schema[1] # Is it better to put it into kwargs? e.g. op_schema.kwargs_schema['dim']
    #print(f"{softmax_dim}, {dim_map}")
    if (softmax_dim < len(dim_map) and dim_map[softmax_dim] >= 0):
        raise RuntimeError(
            "Cannot run softmax on batch dim!"
        )
    return OutputSharding(op_schema.args_spec[0])

@register_prop_rule("aten._softmax_backward_data.default")
def softmax_bwd_rule(op_schema: OpSchema) -> OutputSharding:
    return OutputSharding(op_schema.args_spec[0])
