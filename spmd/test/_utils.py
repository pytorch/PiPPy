# Copyright (c) Meta Platforms, Inc. and affiliates
import sys
from typing import Any, Callable, Tuple, Dict, Optional

import torch
import torch.distributed as dist

from functools import wraps
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
)

TEST_GPU_NUM = 4


class DistTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return TEST_GPU_NUM

    def init_pg(self, backend: str = "nccl") -> None:
        if backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if backend not in ["nccl", "gloo", "mpi"]:
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
        )

        # set device for nccl pg for collectives
        if backend == "nccl":
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()


# wrapper to initialize comms (processgroup)
def with_comms(
    func: Optional[  # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        Callable
    ] = None,
    backend: Optional[str] = None,
) -> Optional[  # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    Callable
]:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore
    ) -> None:
        # if backend not specified, and cuda available, then use nccl, else gloo
        pg_backend = (
            "nccl" if backend is None and torch.cuda.is_available() else "gloo"
        )
        if pg_backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.device_type = "cuda" if pg_backend == "nccl" else "cpu"
        self.init_pg(backend=pg_backend)
        func(self)  # type: ignore
        self.destroy_pg()

    return wrapper
