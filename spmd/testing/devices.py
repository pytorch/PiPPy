from typing import TypeVar, cast

import torch
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

from spmd import DeviceMesh

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

# We use this as a proxy for "multiple GPUs exist"
NUM_DEVICES = 4
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())

T = TypeVar("T")


def skip_unless_torch_gpu(method: T) -> T:
    """
    Test decorator which skips the test unless there's a GPU available to torch.

    >>> @skip_unless_torch_gpu
    >>> def test_some_method(self) -> None:
    >>>   ...
    """
    return cast(T, skip_if_lt_x_gpu(1)(method))


def build_device_mesh() -> DeviceMesh:
    """
    Build a testing device mesh.
    """
    return DeviceMesh(DEVICE_TYPE, list(range(NUM_DEVICES)))  # type: ignore
