# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List
import torch
import torch.nn as nn
from spmd.tensor import DTensor, Placement, Shard, Replicate, _Partial
from spmd.tensor.device_mesh import get_global_device_mesh, DeviceMesh

torch.__future__.set_overwrite_module_params_on_conversion(True)


# pyre-fixme[3]: Return type must be annotated.
def distribute_tensor(
    tensor: torch.Tensor,
    # pyre-fixme[9]: device_mesh has type `DeviceMesh`; used as `None`.
    device_mesh: DeviceMesh = None,
    # pyre-fixme[9]: placements has type `List[Placement]`; used as `None`.
    placements: List[Placement] = None,
):
    # get default device mesh if there's nothing specified
    device_mesh = (
        get_global_device_mesh() if device_mesh is None else device_mesh
    )
    # convert tensor to the correponding device type if it's not in that device type
    tensor = tensor.to(device_mesh.device_type)
    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    # distribute the tensor according to PlacementSpec
    assert len(placements) == 1, "Only support 1-d placement now"
    for idx, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            assert (
                shard_dim <= tensor.ndim
            ), "Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim}"
            # TODO: handle multi-dim device mesh and last shard
            num_chunks = device_mesh.size()
            assert tensor.size(shard_dim) % num_chunks == 0, (
                f"Only support chunk sharding evenly now, but tensor got "
                f"dimension {shard_dim} of size {tensor.size(shard_dim)}, "
                f"which does not divide number of shards {num_chunks}."
            )
            chunk_size = tensor.size(shard_dim) // num_chunks
            tensor_list = list(tensor.chunk(num_chunks, dim=shard_dim))
            scatter_shape = list(tensor.size())
            scatter_shape[shard_dim] = chunk_size
            local_tensor = device_mesh.scatter(tensor_list)
            dist_tensor = DTensor.from_local(
                local_tensor, device_mesh, placements
            )
        elif isinstance(placement, Replicate) or isinstance(
            placement, _Partial
        ):
            dist_tensor = DTensor.from_local(tensor, device_mesh, placements)
        else:
            raise RuntimeError("Not supported!")

    # pyre-fixme[61]: `dist_tensor` is undefined, or not always defined.
    return dist_tensor


# pyre-fixme[3]: Return type must be annotated.
def distribute_module(
    mod: nn.Module,
    # pyre-fixme[9]: device_mesh has type `DeviceMesh`; used as `None`.
    device_mesh: DeviceMesh = None,
    # pyre-fixme[9]: spec has type `List[Placement]`; used as `None`.
    spec: List[Placement] = None,
):
    """
    this function coverts all module parameters
    to distributed tensor parameters according to
    the placements and device_mesh spcified.
    TODO: add a more flexible tagging, i.e. convert
    certain param to a certain spec, like a PlacementPlan
    """

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def to_dist_tensor(t):
        if isinstance(t, nn.Parameter):
            return distribute_tensor(t.data, device_mesh, spec)
        else:
            return t

    mod._apply(to_dist_tensor)

    return mod
