# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Optional, Sequence, cast

import torch
from spmd.tensor.api import DTensor
from spmd.tensor.device_mesh import DeviceMesh, get_global_device_mesh
from spmd.tensor.placement_types import Placement, Shard, Replicate, _Partial


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
        tensor (torch.Tensor): torch.Tensor to be distributed
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
    device_mesh = get_global_device_mesh() if device_mesh is None else device_mesh
    # convert tensor to the correponding device type if it's not in that device type
    tensor = tensor.to(device_mesh.device_type)
    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    # distribute the tensor according to PlacementSpec
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            placement = cast(Shard, placement)
            shard_dim = placement.dim
            assert (
                shard_dim <= tensor.ndim
            ), f"Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim}"
            # TODO: handle uneven shard sizes
            num_chunks = device_mesh.size(dim=idx)
            assert tensor.size(shard_dim) % num_chunks == 0, (
                f"Only support chunk sharding evenly now, but tensor got "
                f"dimension {shard_dim} of size {tensor.size(shard_dim)}, "
                f"which does not divide number of shards {num_chunks}."
            )
            chunk_size = tensor.size(shard_dim) // num_chunks
            tensor_list = list(tensor.chunk(num_chunks, dim=shard_dim))
            scatter_shape = list(tensor.size())
            scatter_shape[shard_dim] = chunk_size
            local_tensor = device_mesh.scatter(tensor_list, mesh_dim=idx)
            # scatter call could not return a tensor with correct requires_grad
            # field, as ProcessGroupNCCL refuse to take a tensor with requires_grad
            # to do inplace update! So we manually set it here
            local_tensor.requires_grad_(tensor.requires_grad)
            tensor = local_tensor
        elif placement.is_replicate():
            tensor = device_mesh.broadcast(tensor, mesh_dim=idx)
        else:
            raise RuntimeError("Not supported!")

    return DTensor(
        tensor,
        device_mesh,
        placements,
        requires_grad=tensor.requires_grad,
    )
