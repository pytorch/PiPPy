# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import spmd.tensor.api as spmd_tensor
from spmd.tensor.placement_types import PlacementSpec
from spmd.tensor.dispatch import OutputSpecType


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def all_equal(xs):
    xs = list(xs)
    if not xs:
        return True
    return xs[1:] == xs[:-1]


def unwrap_local_tensor(e: "spmd_tensor.DTensor") -> torch.Tensor:
    return e._local_tensor if isinstance(e, spmd_tensor.DTensor) else e


def unwrap_spec(e: "spmd_tensor.DTensor") -> PlacementSpec:
    if not isinstance(e, spmd_tensor.DTensor):
        return None
    return e._placement_spec


def wrap(res: object, spec: OutputSpecType) -> object:
    if isinstance(res, torch.Tensor):
        assert spec is not None and isinstance(
            spec, PlacementSpec
        ), "output spec does not match with output!"
        return spmd_tensor.DTensor(res, spec.mesh, spec.placements)
    elif isinstance(res, list):
        assert spec is not None and isinstance(
            spec, list
        ), "output spec does not match with output!"
        return list(
            spmd_tensor.DTensor(e, s.mesh, s.placements)
            for e, s in zip(res, spec)
        )
    elif isinstance(res, tuple):
        assert spec is not None and isinstance(
            spec, tuple
        ), "output spec does not match with output!"
        return tuple(
            spmd_tensor.DTensor(e, s.mesh, s.placements)
            for e, s in zip(res, spec)
        )
    else:
        # if the res contains only non tensor values, we simply return it without rewrapping
        return res
