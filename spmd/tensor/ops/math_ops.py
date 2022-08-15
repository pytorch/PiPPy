# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, Dict
from spmd.tensor.api import DTensor
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.placement_types import Replicate, _Partial, PlacementSpec
from spmd.tensor.ops.utils import register_impl


def _gen_placement_spec_with_pending_sum(
    spec: PlacementSpec, pending_sum: List[int]
) -> PlacementSpec:
    new_placements = [
        placement if i not in pending_sum else _Partial()
        for i, placement in enumerate(spec.placements)
    ]
    return PlacementSpec(spec.ndim, spec.mesh, new_placements)


def einop_rule(
    equation: str,
    op_schema: OpSchema,
    linearity: bool = False,
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
            _gen_placement_spec_with_pending_sum(spec, pending_sums)
            for spec in input_specs
        )
        return OutputSharding(
            None,
            schema_suggestions=[OpSchema(new_arg_specs, {})],
            failed_reason="Input placements does not satisfy linearity property of the op!",
        )

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
                # TODO: merge the sharding properly, if cann't be merged, return suggestion
                assert (
                    dim_to_sharding[dim] == mesh_dim
                ), f"{equation}: dim {dim} sharded two different ways: {mesh_dim} and {dim_to_sharding[dim]}"

    for dim, shard_on_mesh in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)

    return OutputSharding(
        PlacementSpec.from_dim_map(
            input_specs[0].mesh,
            [dim_to_sharding[dim] for dim in output_dim],
            pending_sums,
        )
    )


@register_impl("aten.sum.default")
def dist_sum(self: DTensor) -> DTensor:
    self_local = self.to_local()
    self_placement = self.placements[0]
    device_mesh = self.device_mesh

    local_sum = self_local.sum()

    if self_placement.is_shard() or self_placement.is_partial():
        placements = [_Partial()]
        # partial reduce
        partial_sum = DTensor(local_sum, device_mesh, placements)
        # all_reduce across device
        replicate_placements = [Replicate()]
        return partial_sum.redistribute(device_mesh, replicate_placements)
    elif self_placement.is_replicate():
        return DTensor(
            local_sum, device_mesh=device_mesh, placements=self.placements
        )
    else:
        raise RuntimeError("Not supported!")
