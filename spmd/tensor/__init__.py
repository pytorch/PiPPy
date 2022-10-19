# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Optional, Sequence, cast

import torch
from spmd.tensor.api import DTensor
from spmd.tensor.device_mesh import DeviceMesh, get_global_device_mesh
from spmd.tensor.placement_types import Placement, Shard, Replicate


# Import all builtin dist tensor ops
import spmd.tensor.ops


def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Distribute a torch.Tensor to the `device_mesh` according to the `placements`
    specified. The rank of `device_mesh` and `placements` must be the same.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use `torch.tensor_split`
            semantic to shard the tensor and scatter the shards.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as `device_mesh.ndim`. If not specified, we will
            by default replicate the tensor across the `device_mesh` from the
            first rank of each dimension of the `device_mesh`.

    Returns:
        A :class:`DTensor` object
    """
    # get default device mesh if there's nothing specified
    device_mesh = (
        get_global_device_mesh() if device_mesh is None else device_mesh
    )
    # convert tensor to the correponding device type if it's not in that device type
    tensor = tensor.to(device_mesh.device_type)
    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    if len(placements) != device_mesh.ndim:
        raise ValueError(
            f"`placements` must have the same length as `device_mesh.ndim`! "
            f"Found placements length: {len(placements)}, and device_mesh.ndim: {device_mesh.ndim}."
        )

    if isinstance(tensor, DTensor):
        # if the tensor is already a DTensor, we just need to check if the
        # device mesh and placements are the same
        if tensor.device_mesh != device_mesh:
            raise ValueError(
                f"Cannot distribute a DTensor with device mesh {tensor.device_mesh} "
                f"to a different device mesh {device_mesh}."
            )
        if tensor.placements != placements:
            raise ValueError(
                f"Cannot distribute a DTensor with placements {tensor.placements} "
                f"to a different placements {placements}. do you want to call "
                f"`redistribute` instead?"
            )
        return tensor

    local_tensor = tensor

    # distribute the tensor according to the placements.
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            placement = cast(Shard, placement)

            my_coordinate = device_mesh.get_coordinate_on_dim(idx)
            # TODO: what should happen if rank is not in the mesh?
            # see issue https://github.com/pytorch/tau/pull/492
            assert (
                my_coordinate is not None
            ), "Rank if not part of mesh"  # TODO: figure out behavior here

            num_chunks = device_mesh.size(idx)
            scatter_list, pad_idx = placement.shard_tensor(
                local_tensor, num_chunks, with_padding=True, contiguous=True
            )
            output = torch.empty_like(scatter_list[my_coordinate])

            device_mesh.scatter(output, scatter_list, mesh_dim=idx)
            if pad_idx != 0 and my_coordinate >= pad_idx:
                output = placement.unpad_tensor(output)
            # scatter call could not return a tensor with correct requires_grad
            # field, as ProcessGroupNCCL refuse to take a tensor with requires_grad
            # to do inplace update! So we manually set it here
            output.requires_grad_(tensor.requires_grad)
            local_tensor = output
        elif placement.is_replicate():
            local_tensor = local_tensor.contiguous()
            device_mesh.broadcast(local_tensor, mesh_dim=idx)
        else:
            raise RuntimeError(
                f"Trying to distribute tensor with unsupported placements {placement} on device mesh dimension {idx}!"
            )

    assert local_tensor is not None, "distributing a tensor should not be None"
    return DTensor(
        local_tensor,
        device_mesh,
        placements,
        size=tensor.size(),
        requires_grad=tensor.requires_grad,
    )


# All public APIs from dtensor package
__all__ = [
    "DTensor",
    "DeviceMesh",
    "distribute_tensor",
    "Shard",
    "Replicate",
]
