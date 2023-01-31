# Copyright (c) Meta Platforms, Inc. and affiliates
# from contextlib import contextmanager
from copy import deepcopy

# from dataclasses import dataclass
from functools import wraps
from typing import Literal, Union

import torch
import torch.nn as nn
from spmd.compiler.api import Schema, SPMD
from spmd.compiler.graph_optimization import GraphOptimization
from spmd.tensor import DeviceMesh, Replicate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)

# import spmd.compiler.graph_optimization as graph_optimization


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


class CommOverlapTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_optimizations(self):
        # test grads with overlap pass vs without overlap
        # and vs ddp grads
        model_layer_size = 4
        gpu_placement = torch.arange(self.world_size)

        class OverlapModel(nn.Module):
            def __init__(
                self, layer_count: int = 4, _with_bias: bool = False
            ) -> None:
                super().__init__()

                self.seq = nn.Sequential(
                    *[
                        nn.Linear(10, 10, bias=_with_bias)
                        for _ in range(layer_count)
                    ]
                )

            def forward(
                self, x: torch.Tensor
            ) -> Union[torch.Tensor, Literal[0]]:
                return sum([self.seq(x)])

        base_model = OverlapModel(layer_count=model_layer_size).to(
            self.device_type
        )

        all_spmd = []
        optimizations = [
            [],
            [GraphOptimization("fuse_communication_cat")],
            [GraphOptimization("fuse_communication_jit")],
        ]
        for optim in optimizations:
            spmd = SPMD(
                deepcopy(base_model).to(self.device_type),
                schema=Schema(
                    mesh=DeviceMesh(self.device_type, gpu_placement),
                    placements=[Replicate()],
                ),
                optimize_first_iter=True,
                optimizations=optim,
            )
            all_spmd.append(spmd)
        ddp_model = DDP(deepcopy(base_model)).to(self.device_type)

        input = torch.randn(2, 10).to(self.device_type)
        for i, spmd in enumerate(all_spmd):
            spmd(input).sum().backward()
            if i == 0:
                self.assertFalse(spmd._graph_optimization.optimized, f"{i}")
            else:
                self.assertTrue(spmd._graph_optimization.optimized)
                self.assertFalse(spmd._graph_optimization._optimizing)
        ddp_model(input).sum().backward()

        # compare all_spmd vs DDP
        for spmd in all_spmd:
            for i, (p1, p2) in enumerate(
                zip(ddp_model.parameters(), spmd.parameters())
            ):

                assert p1.grad.allclose(  # type: ignore
                    p2.grad / self.world_size
                ), "Mismatch in resulting grads between DDP and SPMD."

                self.assertTrue(
                    p1.grad.allclose(p2.grad / self.world_size)
                    or p1.grad.allclose(p2.grad)
                )


if __name__ == "__main__":
    run_tests()
