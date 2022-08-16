# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, cast

import torch
import spmd.tensor.api as spmd_tensor
from spmd.tensor.placement_types import Placement, _Partial, Shard, Replicate
from spmd.tensor.device_mesh import DeviceMesh


def _replicate_then_shard(val):
    """
    Replicate from inner to outer dimension.
    Shard from outer to inner dimension.
    """
    i, (current, target) = val
    if (target.is_replicate() or target.is_partial()) and current.is_shard():
        return -i
    elif (current.is_replicate() or current.is_partial()) and target.is_shard():
        return i
    else:
        return 0


def _decompose_reshard(val):
    """
    Decompose Si -> Sj into Si -> R -> Sj
    There's 2 ways a shardings can differ within a mesh dimension:
      1) sharding on different tensor dimensions, e.g. Shard(0) -> Shard(1)
      2) different sub-shards of a repeated shard ("mis-aligned sharding")
          (Shard(0), Shard(0)) -> (Replicate(), Shard(0))
          Here the Shard(0) -> Shard(0) for mesh dimension 2 is actually
          a reshard, because in the first case it's a sub-sharding of an already tensor dimension 0,
          and in the second case, it's the first sharding on tensor dimesnion 0.
    """
    # detect mis-aligned repeated shardings
    from collections import defaultdict
    repeat_dim_current = defaultdict(int)
    repeat_dim_target = defaultdict(int)

    output = []

    for i, (current, target) in val:
        # detect mis-aligned sharding
        if current.is_shard():
            repeat_dim_current[current.dim] += 1
        if target.is_shard():
            repeat_dim_target[target.dim] += 1
        if (
                current.is_shard() and target.is_shard() and
                (current.dim != target.dim or repeat_dim_current[current.dim] != repeat_dim_target[target.dim])
        ):
            # decompose Shard(i) -> Shard(j) into Shard(i) -> Replicate() -> Shard(j)
            output.append((i, (current, Replicate())))
            output.append((i, (Replicate(), target)))
        else:
            output.append((i, (current, target)))

    return output


def redistribute_spmd_tensor(
    input: "spmd_tensor.DTensor",
    device_mesh: DeviceMesh,
    placements: List[Placement],
) -> "spmd_tensor.DTensor":
    current_placements = input.placements
    local_tensor = input.to_local()
    if input.device_mesh != device_mesh:
        # TODO: alltoall reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = None

    sorted_placements = list(enumerate(zip(current_placements, placements)))
    sorted_placements = _decompose_reshard(sorted_placements)
    sorted_placements.sort(key=_replicate_then_shard)
    attempted_transforms = [None] * len(placements)

    for i, (current, target) in sorted_placements:
        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            attempted_transforms[i] = target
            continue

        assert (
            not target.is_partial()
        ), "Cannot create partial via redistribute!"

        if target.is_replicate():
            # Case 1: target is Replicate
            attempted_transforms[i] = target
            if current.is_partial():
                partial_spec = cast(_Partial, current)
                # all_reduce
                new_local_tensor = device_mesh.all_reduce(
                    local_tensor, partial_spec.reduce_op, mesh_dim=i
                )
            else:
                assert current.is_shard()
                # for shard, all_gather all shards and return a tensor that is replicated on the previously sharded dimension
                shard_spec = cast(Shard, current)
                new_size = list(local_tensor.size())
                new_size[shard_spec.dim] *= device_mesh.size(
                    i
                )  # only works for evenly sharded tensors
                new_local_tensor = torch.empty(
                    new_size, device=local_tensor.device, dtype=input.dtype
                )
                # NOTE: all_gather_base only works well when tensor
                # sharded on a sequential list of devices
                device_mesh.all_gather_base(
                    new_local_tensor,
                    local_tensor,
                    mesh_dim=i,
                    tensor_dim=shard_spec.dim,
                )
        else:
            assert target.is_shard()
            # Case 2: target is Shard

            shard_dim = target.dim  # type: ignore
            num_chunks = device_mesh.size(dim=i)
            assert (
                input.size(shard_dim) % num_chunks == 0
            ), "Only support chunk sharding evenly now"
            chunk_size = (
                local_tensor.size(shard_dim) // num_chunks
            )  # this may already be sharded on a differernt mesh dimension
            my_rank = device_mesh.get_coordinate_on_dim(dim=i)
            assert (
                my_rank is not None
            ), "Rank is not part of the mesh"  # TODO: figure out behavior here
            if current.is_partial():
                # reduce scatter the current tensors
                attempted_transforms[i] = target
                new_tensor_size = list(local_tensor.size())
                new_tensor_size[shard_dim] = chunk_size
                new_local_tensor = torch.empty(
                    new_tensor_size,
                    device=local_tensor.device,
                    dtype=input.dtype,
                )
                new_local_tensor = device_mesh.reduce_scatter_base(
                    new_local_tensor, local_tensor, mesh_dim=i
                )
            elif current.is_replicate():
                attempted_transforms[i] = target
                # slice/narrow the tensor to corresponding local shard then return shard tensor
                new_local_tensor = local_tensor.narrow(
                    shard_dim, my_rank * chunk_size, chunk_size
                )

        assert new_local_tensor is not None
        local_tensor = new_local_tensor

    if attempted_transforms != list(placements):
        # TODO: if not the same, we should apply all_to_all reshuffle
        raise NotImplementedError(f"Could not redistribute the tensor! Got {attempted_transforms} vs {list(placements)}")

    assert new_local_tensor is not None, "redistribute failed!"

    return spmd_tensor.DTensor(new_local_tensor, device_mesh, placements)


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "spmd_tensor.DTensor",
        device_mesh: DeviceMesh,
        placements: List[Placement],
    ):
        ctx.previous_placement = input.placements
        ctx.previous_device_mesh = input.device_mesh
        return redistribute_spmd_tensor(input, device_mesh, placements)

    @staticmethod
    def backward(ctx, grad_output: "spmd_tensor.DTensor"):  # type: ignore
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh
        return (
            redistribute_spmd_tensor(
                grad_output, previous_device_mesh, previous_placement
            ),
            None,
            None,
        )
