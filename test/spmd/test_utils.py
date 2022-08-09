# Copyright (c) Meta Platforms, Inc. and affiliates
import sys

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
    def world_size(self):
        return TEST_GPU_NUM

    def init_pg(self, backend="nccl"):
        if backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if backend not in ["nccl", "gloo", "mpi"]:
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # set device for nccl pg for collectives
        if backend == "nccl":
            torch.cuda.set_device(self.rank)

    def destroy_pg(self):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()


# wrapper to initialize comms (processgroup)
def with_comms(func=None, backend=None):
    assert func is not None

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # if backend not specified, and cuda available, then use nccl, else gloo
        pg_backend = (
            "nccl" if backend is None and torch.cuda.is_available() else "gloo"
        )
        if pg_backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.device_type = "cuda" if pg_backend == "nccl" else "cpu"
        self.init_pg(backend=pg_backend)
        func(self)
        self.destroy_pg()

    return wrapper
