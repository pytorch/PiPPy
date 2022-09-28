# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import Dict, Tuple, Union

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

import spmd.tensor.api as dtensor
from spmd.tensor.placement_types import DTensorSpec, OutputSpecType
from spmd.tensor.redistribute import redistribute_dtensor

ArgKwargsType = Union[Tuple[object, ...], Dict[str, object]]

# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def all_equal(xs):
    xs = list(xs)
    if not xs:
        return True
    return xs[1:] == xs[:-1]


def unwrap_local_tensor(e: "dtensor.DTensor") -> torch.Tensor:
    return e._local_tensor if isinstance(e, dtensor.DTensor) else e


def unwrap_schema(e: object) -> object:
    return e._spec if isinstance(e, dtensor.DTensor) else e


def wrap(res: object, spec: OutputSpecType) -> object:
    if isinstance(res, torch.Tensor):
        assert spec is not None and isinstance(
            spec, DTensorSpec
        ), f"output spec does not match with output! Expected DTensorSpec, got {spec}."
        return dtensor.DTensor(res, spec.mesh, spec.placements)
    elif isinstance(res, list):
        assert spec is not None and isinstance(
            spec, list
        ), f"output spec does not match with output! Expected list, got {spec}."
        return list(
            dtensor.DTensor(e, s.mesh, s.placements)
            for e, s in zip(res, spec)
        )
    elif isinstance(res, tuple):
        assert spec is not None and isinstance(
            spec, tuple
        ), f"output spec does not match with output! Expected tuple, got {spec}"
        return tuple(
            dtensor.DTensor(e, s.mesh, s.placements)
            for e, s in zip(res, spec)
        )
    else:
        # if the res contains only non tensor values, we simply return it without rewrapping
        return res


def pack_args_kwargs_with_local_tensor(
    args: ArgKwargsType,
    args_schema: ArgKwargsType,
    redistribute_with_schema: bool = False,
) -> ArgKwargsType:
    flatten_args, args_tree_spec = tree_flatten(args)
    flatten_args_schema, _ = tree_flatten(args_schema)

    for i, arg in enumerate(flatten_args):
        if isinstance(arg, dtensor.DTensor):
            if redistribute_with_schema:
                target_spec = flatten_args_schema[i]
                arg = redistribute_dtensor(
                    arg, target_spec.mesh, target_spec.placements
                )

            # reuse the schema list and update it with local tensor
            flatten_args_schema[i] = arg._local_tensor

    return tree_unflatten(flatten_args_schema, args_tree_spec)
