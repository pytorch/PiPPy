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


def _register_element_wise_op(op_name, op):
    @register_impl(op_name)
    def _dist_element_wise(*args, **kwargs) -> Tensor:
        self = args[0]
        self_placement = self.placements[0]
        if self_placement.is_partial():
            raise RuntimeError("Not supported!")
        else:
            print("DDDDDDDDD")
            args = (self.local_tensor(), *args[1:])
            return Tensor.from_local(
                op(*args, **kwargs), device_mesh=self.device_mesh, placements=self.placements
            )