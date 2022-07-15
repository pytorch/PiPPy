# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List
import torch
import torch.nn as nn
from spmd.tensor import (
    Tensor,
    Placement,
    Shard,
    Replicate,
    _Partial
)
from spmd.tensor.device_mesh import (
    get_global_device_mesh,
    DeviceMesh,
)

torch.__future__.set_overwrite_module_params_on_conversion(True)


def distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh = None, placements: List[Placement] = None):
    # get default device mesh if there's nothing specified
    device_mesh = get_global_device_mesh() if device_mesh is None else device_mesh
    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in device_mesh.ndim]

    # distribute the tensor according to PlacementSpec
    assert len(placements) == 1, "Only support 1-d placement now"
    for idx, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            assert shard_dim <= tensor.ndim, "Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim}"
            # TODO: handle multi-dim device mesh and last shard
            num_chunks = device_mesh.size()
            assert tensor.size(shard_dim) % num_chunks == 0, "Only support chunk sharding evenly now"
            chunk_size = tensor.size(shard_dim) // num_chunks
            tensor_list = list(tensor.chunk(num_chunks, dim=shard_dim))
            scatter_shape = list(tensor.size())
            scatter_shape[shard_dim] = chunk_size
            local_tensor = device_mesh.scatter(tensor_list)
            dist_tensor = Tensor.from_local(local_tensor, device_mesh, placements)
        elif isinstance(placement, Replicate) or isinstance(placement, _Partial):
            dist_tensor = Tensor.from_local(tensor, device_mesh, placements)
        else:
            raise RuntimeError("Not supported!")

    return dist_tensor


def distribute_module(mod: nn.Module, device_mesh: DeviceMesh = None, spec: List[Placement] = None):
    '''
    this function coverts all module parameters
    to distributed tensor parameters according to
    the placements and device_mesh spcified.
    TODO: add a more flexible tagging, i.e. convert
    certain param to a certain spec, like a PlacementPlan
    '''
    def to_dist_tensor(t):
        if isinstance(t, nn.Parameter):
            return distribute_tensor(t.data, device_mesh, spec)
        else:
            return t

    mod._apply(to_dist_tensor)

    return mod
