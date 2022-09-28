# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Callable, List, Optional, Union

from spmd.tensor import Placement
from spmd.tensor.api import DTensor


def unwrap_single_placement(e) -> Optional[Placement]:
    if not isinstance(e, DTensor):
        return None
    assert len(e.placements) == 1, "more than one placement!"
    return e.placements[0]


# convenient wrapper to register custom operator impls
def register_impl(func: str) -> Callable:
    def wrapper(impl: Callable):
        DTensor._custom_dispatch_ops[func] = impl
        return impl

    return wrapper


# convenient wrapper to register sharding propagation rules
def register_prop_rule(func: str) -> Callable:
    def wrapper(impl: Callable):
        DTensor._op_to_rules[func] = impl
        return impl

    return wrapper


def as_list(x: Union[List[object], object]) -> List[object]:
    if type(x) is list:
        return x
    else:
        return [x]
