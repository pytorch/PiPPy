# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import Tensor


# convenient wrapper to register custom operator impls
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def register_impl(func):
    # pyre-fixme[53]: Captured variable `func` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def wrapper(impl):
        Tensor._custom_dispatch_ops[func] = impl
        return impl

    return wrapper

def register_prop_rule(func):
    # pyre-fixme[53]: Captured variable `func` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def wrapper(impl):
        Tensor._op_to_rules[func] = impl
        return impl

    return wrapper
