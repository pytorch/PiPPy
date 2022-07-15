# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import Tensor
from spmd.tensor.placement_types import Shard


def unwrap_single_placement(e):
    if not isinstance(e, Tensor):
        return None
    assert len(e.placements) == 1, "more than one placement!"
    return e.placements[0]


def unwrap_local_tensor(e):
    if not isinstance(e, Tensor):
        return None
    return e.local_tensor()


def is_shard_on_dim(placement, dim):
    return isinstance(placement, Shard) and placement.dim == dim

# convenient wrapper to register functions


def register_impl(func):
    def wrapper(impl):
        Tensor._dist_tensor_dispatch_ops[func] = impl
        return impl

    return wrapper
