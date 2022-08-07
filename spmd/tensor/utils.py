# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Sequence

import torch
import spmd.tensor.api as spmd_tensor
from spmd.tensor.device_mesh import DeviceMesh
from spmd.tensor.placement_types import Placement, PlacementSpec


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def all_equal(xs):
    xs = list(xs)
    if not xs:
        return True
    return xs[1:] == xs[:-1]


def unwrap_local_tensor(e: "spmd_tensor.DTensor") -> torch.Tensor:
    return e._local_tensor if isinstance(e, spmd_tensor.DTensor) else e


def unwrap_placements(e: "spmd_tensor.DTensor") -> Sequence[Placement]:
    return (
        e._placement_spec.placements
        if isinstance(e, spmd_tensor.DTensor)
        else e
    )


def unwrap_mesh(e: "spmd_tensor.DTensor") -> DeviceMesh:
    # if this tensor is not Distributed, then return none. We will reinterpret it as replicated
    if not isinstance(e, spmd_tensor.DTensor):
        return None
    mesh = e._placement_spec.mesh
    assert (
        mesh.ndim == 1
    ), "DistributedTensor ops not supporting multi-dim mesh yet"
    return mesh


def unwrap_spec(e: "spmd_tensor.DTensor") -> PlacementSpec:
    if not isinstance(e, spmd_tensor.DTensor):
        return None
    return e._placement_spec


def wrap(e: torch.Tensor, spec: PlacementSpec) -> "spmd_tensor.DTensor":
    return (
        spmd_tensor.DTensor.from_local(
            e, spec.mesh, spec.placements, run_check=False
        )
        if isinstance(e, torch.Tensor)
        else e
    )
