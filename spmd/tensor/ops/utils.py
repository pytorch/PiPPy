# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import Tensor


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def unwrap_single_placement(e):
    if not isinstance(e, Tensor):
        return None
    assert len(e.placements) == 1, "more than one placement!"
    return e.placements[0]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def unwrap_local_tensor(e):
    if not isinstance(e, Tensor):
        return None
    return e.local_tensor()


# convenient wrapper to register functions
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def register_impl(func):
    # pyre-fixme[53]: Captured variable `func` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def wrapper(impl):
        Tensor._dist_tensor_dispatch_ops[func] = impl
        return impl

    return wrapper
