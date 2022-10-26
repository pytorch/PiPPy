# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, List, Sequence, Tuple, cast

import torch
from torch.distributed._spmd.comm_tensor import CommTensor
import spmd.tensor.api as dtensor
from spmd.tensor.placement_types import Placement, _Partial, Shard, Replicate
from spmd.tensor.device_mesh import DeviceMesh


_PlacementItem = Tuple[int, Tuple[Placement, Placement]]


def _replicate_then_shard(val: _PlacementItem) -> int:
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


def _decompose_reshard(val: List[_PlacementItem]) -> List[_PlacementItem]:
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

    repeat_dim_current: Dict[int, int] = defaultdict(int)
    repeat_dim_target: Dict[int, int] = defaultdict(int)

    output: List[_PlacementItem] = []

    for i, (current, target) in val:
        # detect mis-aligned sharding
        if current.is_shard():
            repeat_dim_current[cast(Shard, current).dim] += 1
        if target.is_shard():
            repeat_dim_target[cast(Shard, target).dim] += 1
        if (
            isinstance(current, Shard)
            and isinstance(target, Shard)
            and (
                current.dim != target.dim
                or repeat_dim_current[current.dim]
                != repeat_dim_target[target.dim]
            )
        ):
            # decompose Shard(i) -> Shard(j) into Shard(i) -> Replicate() -> Shard(j)
            output.append((i, (current, Replicate())))
            output.append((i, (Replicate(), target)))
        else:
            output.append((i, (current, target)))

    return output


# Intentionally expose this API to trace ops on local tensors
def _redistribute_with_local_tensor(
    local_tensor: torch.Tensor,
    size: torch.Size,
    device_mesh: DeviceMesh,
    current_placements: Sequence[Placement],
    target_placements: Sequence[Placement],
) -> torch.Tensor:
    new_local_tensor = None

    sorted_placements = list(
        enumerate(zip(current_placements, target_placements))
    )
    sorted_placements = _decompose_reshard(sorted_placements)
    sorted_placements.sort(key=_replicate_then_shard)

    for i, (current, target) in sorted_placements:
        my_coordinate = device_mesh.get_coordinate_on_dim(i)
        num_chunks = device_mesh.size(dim=i)
        # TODO: what should happen if rank is not in the mesh?
        # see issue https://github.com/pytorch/tau/pull/492
        assert (
            my_coordinate is not None
        ), "Rank if not part of mesh"  # TODO: figure out behavior here

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(_Partial, current)
                # out-of-place all_reduce to replicate, since the current partial DTensor
                # might get used by other ops as well, so we can't inplace modify it
                cloned_local = CommTensor(
                    local_tensor.clone(memory_format=torch.contiguous_format)
                )
                device_mesh.all_reduce(
                    cloned_local, partial_spec.reduce_op, mesh_dim=i
                )
                new_local_tensor = cloned_local
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, size, device_mesh, i
                )
            else:
                raise RuntimeError(
                    f"redistribute from {current_placements} to {target_placements} not supported yet"
                )
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                # reduce shard (i.e. reduce_scatter) the current tensors
                new_local_tensor = target_placement._reduce_shard_tensor(
                    local_tensor, device_mesh, i
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                shards, _ = target_placement._split_tensor(
                    local_tensor,
                    num_chunks,
                    with_padding=False,
                    contiguous=False,
                )
                new_local_tensor = shards[my_coordinate].clone()
            else:
                # NOTE: this case shouldn't hit _decompose_sharding, decompose sharding should
                # decompose Shard(0) -> Shard(1) into Shard(0) -> Replicate -> Shard(1)
                assert (
                    current.is_shard()
                ), f"Current placement should be shard but found {current}"
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    # TODO: enable this with all_to_all
                    raise NotImplementedError(
                        "Changing sharding dim is not supported yet!"
                    )

        elif target.is_partial():
            if current.is_replicate():
                # For replicate -> partial, we zero out all other ranks of the current mesh dim
                # and leave only 1 rank have the data, to perform a "zero cost" reshard.
                if my_coordinate is not None and my_coordinate != 0:
                    new_local_tensor = local_tensor.zero_()
                else:
                    new_local_tensor = local_tensor
            else:
                raise RuntimeError(
                    f"redistribute from {current_placements} to {target_placements} not supported yet"
                )

        assert new_local_tensor is not None
        local_tensor = new_local_tensor

    assert new_local_tensor is not None, "redistribute failed!"

    return new_local_tensor


def redistribute_dtensor(
    input: "dtensor.DTensor",
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> "dtensor.DTensor":
    if input.device_mesh != device_mesh:
        # TODO: alltoall reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    local_tensor = input._local_tensor
    new_local_tensor = _redistribute_with_local_tensor(
        local_tensor,
        input.size(),
        device_mesh,
        input.placements,
        placements,
    )

    return dtensor.DTensor(
        new_local_tensor,
        device_mesh,
        placements,
        size=input.size(),
        requires_grad=local_tensor.requires_grad,
    )


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "dtensor.DTensor",
        device_mesh: DeviceMesh,
        placements: List[Placement],
    ):
        ctx.previous_placement = input.placements
        ctx.previous_device_mesh = input.device_mesh
        return redistribute_dtensor(input, device_mesh, placements)

    @staticmethod
    def backward(ctx, grad_output: "dtensor.DTensor"):  # type: ignore
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
            redistribute_dtensor(
                grad_output, previous_device_mesh, target_placements
            ),
            None,
            None,
        )
