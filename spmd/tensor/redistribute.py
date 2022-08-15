# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, cast

import torch
import spmd.tensor.api as spmd_tensor
from spmd.tensor.placement_types import Placement, _Partial
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

    assert len(placements) == 1, "Only support 1-d placement for now"

    attempted_transforms = []
    new_local_tensor = None
    for i, (current, target) in enumerate(zip(current_placements, placements)):
        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            attempted_transforms.append(target)
            continue

        if target.is_replicate():
            # Case 1: target is Replicate
            attempted_transforms.append(target)
            if current.is_partial():
                partial_spec = cast(_Partial, current)
                # all_reduce
                new_local_tensor = device_mesh.all_reduce(
                    local_tensor, partial_spec.reduce_op
                )
            else:
                # for shard, all_gather all shards and return the global tensor
                new_local_tensor = torch.empty(
                    input.size(), device=local_tensor.device, dtype=input.dtype
                )
                # NOTE: all_gather_base only works well when tensor
                # sharded on a sequential list of devices
                device_mesh.all_gather_base(new_local_tensor, local_tensor)
        elif target.is_shard():
            # Case 2: target is Shard
            assert target.is_shard()
            shard_dim = target.dim  # type: ignore
            num_chunks = device_mesh.size(i)
            assert (
                input.size(shard_dim) % num_chunks == 0
            ), "Only support chunk sharding evenly now"
            chunk_size = input.size(shard_dim) // num_chunks
            my_rank = device_mesh.get_rank()
            if current.is_partial():
                # reduce scatter the current tensors
                attempted_transforms.append(target)
                new_tensor_size = list(input.size())
                new_tensor_size[shard_dim] = chunk_size
                new_local_tensor = torch.empty(
                    new_tensor_size,
                    device=local_tensor.device,
                    dtype=input.dtype,
                )
                new_local_tensor = device_mesh.reduce_scatter_base(
                    new_local_tensor, local_tensor
                )
            elif current.is_replicate():
                attempted_transforms.append(target)
                # slice/narrow the tensor to corresponding local shard then return shard tensor
                # TODO: support multi dim redistribute require to not use get_rank, as it would
                # get global_rank and in the case of multidim mesh it's not correct!
                new_local_tensor = local_tensor.narrow(
                    shard_dim, my_rank * chunk_size, chunk_size
                )
            else:
                # diff shard dim on new placement, record in attempted transforms
                attempted_transforms.append(current)
        elif target.is_partial():
            if current.is_replicate():
                # For replicate -> partial, we zero out all other ranks of the current mesh dim
                # and leave only 1 rank have the data, to perform a "zero cost" reshard.
                attempted_transforms.append(target)
                my_rank = device_mesh.get_rank()
                if my_rank != 0:
                    new_local_tensor = local_tensor.zero_()
                else:
                    new_local_tensor = local_tensor
            else:
                raise RuntimeError(
                    f"redistribute from {current_placements} to {placements} not supported yet"
                )

    if attempted_transforms != placements:
        # TODO: if not the same, we should apply all_to_all reshuffle
        raise NotImplementedError("Reshuffling tensor dims not supported yet!")

    assert new_local_tensor is not None, "redistribute failed!"

    return spmd_tensor.DTensor(
        new_local_tensor,
        device_mesh,
        attempted_transforms,
        requires_grad=new_local_tensor.requires_grad,
    )


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
        # When we run backward pass of redistribute (i.e. manual redistribute from
        # user code instead of torch_dispatch), we scan first and see if we need
        # to change the target placement for one special case:
        #   replicate -> partial.
        # In this case we keep the grad as replicate, this is because we don't
        # want to convert the replicated gradients back to partial, although
        # that's logically conform with the same layout, converting the gradients
        # back to partial is acutally useless as you would have to do reduce later
        # which would be more expensive than keeping it replicate! For this reason,
        # we keep the replicate grad here.
        # TODO: see if this make sense for all cases.
        target_placements: List[Placement] = []
        for current, target in zip(grad_output.placements, previous_placement):
            if current.is_replicate() and target.is_partial():
                # keep target placement to replicate instead of partial in this case
                target_placements.append(current)
            else:
                target_placements.append(target)

        return (
            redistribute_spmd_tensor(
                grad_output,
                previous_device_mesh,
                target_placements,
            ),
            None,
            None,
        )
