from functools import wraps

import torch
import torch.nn as nn
from spmd.compiler.api import Schema, SPMD
from spmd.compiler.graph_optimization import GraphOptimization
from spmd.tensor import DeviceMesh, Replicate
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)


def with_comms(func):
    @base_with_comms
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # make sure we set different random seeds for each rank
        # otherwise we dont need DDP / SPMD
        # (we would have the same parameters and inputs everywhere)
        torch.manual_seed(torch.distributed.get_rank())
        return func(self, *args, **kwargs)

    return wrapper


class TestMisc(DTensorTestBase):
    @with_comms
    def test_print_graph(self) -> None:
        model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)]).to(
            self.device_type
        )
        x = torch.randn(2, 10).to(self.device_type)
        spmd_model = SPMD(
            model,
            schema=Schema(
                mesh=DeviceMesh(
                    self.device_type, torch.arange(self.world_size)
                ),
                placements=[Replicate()],
            ),
            optimize_first_iter=True,
            print_graph=True,
        )
        spmd_model = SPMD(
            model,
            schema=Schema(
                mesh=DeviceMesh(
                    self.device_type, torch.arange(self.world_size)
                ),
                placements=[Replicate()],
            ),
            optimizations=[GraphOptimization("overlap_communication")],
            optimize_first_iter=True,
            print_graph=True,
        )


if __name__ == "__main__":
    run_tests()
