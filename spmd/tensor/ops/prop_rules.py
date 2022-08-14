# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Tuple, List, Dict, Optional
from spmd.tensor.placement_types import _Partial, PlacementSpec


def einop_prop(
    equation: str, input_specs: Tuple[PlacementSpec, ...], linearity: bool = False
) -> Optional[PlacementSpec]:
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
    # parse einop equation
    inputs, outputs = equation.split("->")
    input_dims, output_dims = inputs.split(","), outputs.split(",")
    # NOTE: only support single output unless needed in future
    output_dim = output_dims[0]
    dim_to_sharding: Dict[str, int] = {}
    pending_sums: List[int] = []

    seen_shardings = {}

    for input in input_specs:
        for i, a in enumerate(input.placements):
            if a.is_partial():
                seen_shardings[i] = "+"
                if i not in pending_sums:
                    pending_sums.append(i)
                if not linearity:
                    raise RuntimeError(
                        "cannot do generic op on a tensor with partial sums"
                    )

    for input_dim, input_spec in zip(input_dims, input_specs):
        for dim, mesh_dim in zip(input_dim, input_spec.dim_map):
            if dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
            else:
                # TODO: merge the sharding properly
                assert dim_to_sharding[dim] == mesh_dim, ""

    for dim, shard_on_mesh in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)

    return PlacementSpec.from_dim_map(
        input_specs[0].mesh,
        [dim_to_sharding[dim] for dim in output_dim],
        pending_sums,
    )


def mm_prop(
    mat1_spec: PlacementSpec, mat2_spec: PlacementSpec
) -> Optional[PlacementSpec]:
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
        return PlacementSpec(mat1_spec.ndim, mat1_spec.mesh, placements)
    elif (
        mat1_spec.placements[0].is_replicate()
        and mat2_spec.placements[0].is_replicate()
    ):
        return mat1_spec
    else:
        # not local compute, need to rely on auto redistribute, return None
        return None


def pointwise_prop(
    input_specs: Tuple[PlacementSpec, ...], linearity: bool = False
) -> Optional[PlacementSpec]:
    """
    Propagate the sharding for pointwise operations. Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # handle the case of broadcasting, find the max_dim first
    max_dim = max(input.ndim for input in input_specs)
    dimchars = []
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"
    return einop_prop(fmt, input_specs, linearity=linearity)
