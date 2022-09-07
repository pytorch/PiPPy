# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, Dict, Tuple, cast
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.placement_types import _Partial, DTensorSpec
from spmd.tensor.ops.utils import as_list


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
    # record pending sum, key is mesh dimension, value is pending sum
    # counter across input specs
    pending_sums_counter: Dict[int, int] = {}
    seen_shardings = {}
    needs_reshard = False

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
        # deal with partial sums
        input_sums = input_spec.sums
        for sum_dim in input_sums:
            if sum_dim not in pending_sums_counter:
                seen_shardings[sum_dim] = "+"
            # update pending sum counter for pending sum mesh
            # dimension with the occurance from each input
            pending_sums_counter[sum_dim] = (
                pending_sums_counter.get(sum_dim, 0) + 1
            )

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

    if pending_sums_counter and not linearity:
        raise RuntimeError("Cannot do generic op on a tensor with partial sums")
    else:
        # It's a op that support linearity, but not all input arguments are partial
        # we fail the sharding propagation with suggestion to make all inputs be
        # partial on the corresponding mesh dim (all inputs should be partial for
        # the mesh dims in order to execute locally and delay the sum reduction)
        for value in pending_sums_counter.values():
            if value != len(input_specs):
                needs_reshard = True

    pending_sums = list(pending_sums_counter.keys())
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
