# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Tuple, List, Dict
from torch.distributed.distributed_c10d import ReduceOp
from spmd.tensor.api import DTensor
from spmd.tensor.placement_types import Replicate, _Partial, PlacementSpec
from spmd.tensor.ops.utils import register_impl, register_prop_rule
from spmd.tensor.dispatch import OpInfo


def _parse_einop_equation(equation: str) -> Tuple[List[str], List[str]]:
    inputs, outputs = equation.split("->")
    input_dims = []
    output_dims = []
    for input_dim in inputs.split(","):
        input_dims.append(input_dim)

    for output_dim in outputs.split(","):
        output_dims.append(output_dim)

    return input_dims, output_dims


def einop_prop(equation: str, input_specs: List[PlacementSpec], linear=False):
    """
    Propagate the sharding of inputs to output for ops whose data
    moves according to einsum notation. This is mostly borrowed
    from @zdevito's sharding simulator. Examples:
        mk,kn->mn - einsum
        ij,ij->ij - addition
        ij,j->ij - broadcasted addition
        ij->i - reduction
    Other ops could use this propagation algorithm when applied.
    """
    input_dims, output_dims = _parse_einop_equation(equation)
    # NOTE: only support single output unless needed in future
    output_dim = output_dims[0]
    dim_to_sharding: Dict[str, int] = {}
    pending_sums: List[int] = []

    seen_shardings = {}

    for input in input_specs:
        for i, a in enumerate(input.placements):
            if a.is_partial():
                seen_shardings[i] = "+"
                if not i in pending_sums:
                    pending_sums.append(i)
                if not linear:
                    raise RuntimeError(
                        "cannot do generic op on a tensor with partial sums"
                    )

    for input_dim, input_spec in zip(input_dims, input_specs):
        for dim, mesh_dim in zip(input_dim, input_spec.dims_map):
            if dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
            else:
                # TODO: merge the sharding properly
                assert dim_to_sharding[dim] == mesh_dim, ""

    for dim, shard_on_mesh in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)

    return PlacementSpec.from_dims_map(
        input_specs[0].mesh,
        [dim_to_sharding[dim] for dim in output_dim],
        pending_sums,
    )


@register_impl("aten.sum.default")
def dist_sum(self: DTensor) -> DTensor:
    self_local = self.to_local()
    self_placement = self.placements[0]
    device_mesh = self.device_mesh

    local_sum = self_local.sum()

    if self_placement.is_shard() or self_placement.is_partial():
        placements = [_Partial(ReduceOp.SUM)]
        # partial reduce
        partial_sum = DTensor.from_local(local_sum, device_mesh, placements)
        # all_reduce across device
        replicate_placements = [Replicate()]
        return partial_sum.redistribute(device_mesh, replicate_placements)
    elif self_placement.is_replicate():
        return DTensor.from_local(
            local_sum, device_mesh=device_mesh, placements=self.placements
        )
    else:
        raise RuntimeError("Not supported!")
