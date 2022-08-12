# Copyright (c) Meta Platforms, Inc. and affiliates
from re import M
from typing import List, cast

import torch
import spmd.tensor.api as spmd_tensor
from spmd.tensor.placement_types import Placement, _Partial, Shard
from spmd.tensor.device_mesh import DeviceMesh


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

    attempted_transforms = []
    new_local_tensor = None

    # we need to go backwards in the case of to respect the scatter order established in 
    # distribute_tensor(), for the cases where we're sharding multiple times on the same
    # tensor dimension (e.g. [Shard(0), Shard(0)])
    reverted_placements = list(enumerate(zip(current_placements, placements))) 
    reverted_placements.reverse()

    for i, (current, target) in reverted_placements:
        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            attempted_transforms.append(target)
            continue

        assert (
            not target.is_partial()
        ), "Cannot create partial via redistribute!"

        if target.is_replicate():
            # Case 1: target is Replicate
            attempted_transforms.append(target)
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
                new_size[shard_spec.dim] *= device_mesh.size(i)  # only works for evenly sharded tensors
                new_local_tensor = torch.empty(
                    new_size, device=local_tensor.device, dtype=input.dtype
                )
                # NOTE: all_gather_base only works well when tensor
                # sharded on a sequential list of devices
                device_mesh.all_gather_base(new_local_tensor, local_tensor, mesh_dim=i, tensor_dim=shard_spec.dim)
        else:
            assert target.is_shard()
            # Case 2: target is Shard

            shard_dim = target.dim  # type: ignore
            num_chunks = device_mesh.size(dim=i)
            assert (
                input.size(shard_dim) % num_chunks == 0
            ), "Only support chunk sharding evenly now"
            chunk_size = local_tensor.size(shard_dim) // num_chunks  # this may already be sharded on a differernt mesh dimension
            my_rank = device_mesh.get_rank_for_dim(dim=i)
            assert my_rank is not None, 'Rank is not part of the mesh'   # TODO: figure out behavior here
            if current.is_partial():
                # reduce scatter the current tensors
                attempted_transforms.append(target)
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
                attempted_transforms.append(target)
                # slice/narrow the tensor to corresponding local shard then return shard tensor
                new_local_tensor = local_tensor.narrow(
                    shard_dim, my_rank * chunk_size, chunk_size
                )
            else:
                # diff shard dim on new placement, record in attempted transforms
                # temporary: replicate then shard again
                # TODO : implement with all_to_all instead
                assert current.is_shard()
                assert target.is_shard()

                from_shard_spec = cast(Shard, current)
                to_shard_spec = cast(Shard, target)
                new_size = list(local_tensor.size())
                new_size[from_shard_spec.dim] *= device_mesh.size(i)  # only works for evenly sharded tensors
                new_local_tensor = torch.empty(
                    new_size, device=local_tensor.device, dtype=input.dtype
                )
                new_size[to_shard_spec.dim]
                # NOTE: all_gather_base only works well when tensor
                # sharded on a sequential list of devices
                device_mesh.all_gather_base(new_local_tensor, local_tensor, mesh_dim=i, tensor_dim=from_shard_spec.dim)
                new_local_tensor = new_local_tensor.narrow(
                    to_shard_spec.dim, my_rank * chunk_size, chunk_size
                )
                attempted_transforms.append(target)


        local_tensor = new_local_tensor
 
    attempted_transforms.reverse()
    if attempted_transforms != list(placements):
        # TODO: if not the same, we should apply all_to_all reshuffle
        raise NotImplementedError("Could not redistribute the tensor!")

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
