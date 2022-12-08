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

    @with_comms
    def test_overlap_pass_grads(self):
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
        overlap_model = deepcopy(base_model).to(self.device_type)

        spmd_base = SPMD(
            base_model,
            schema=Schema(
                mesh=DeviceMesh(self.device_type, gpu_placement),
                placements=[Replicate()],
            ),
            optimize_first_iter=True,
        )
        self.assertFalse(spmd_base._graph_optimization.optimized)

        spmd_overlap = SPMD(
            overlap_model,
            schema=Schema(
                mesh=DeviceMesh(self.device_type, gpu_placement),
                placements=[Replicate()],
            ),
            optimize_first_iter=True,
            optimizations=[GraphOptimization("overlap_communication")],
        )
        ddp_model = DDP(deepcopy(overlap_model)).to(self.device_type)

        input = torch.randn(2, 10).to(self.device_type)

        spmd_base(input).sum().backward()
        spmd_overlap(input).sum().backward()
        ddp_model(input).sum().backward()

        self.assertTrue(spmd_overlap._graph_optimization.optimized)
        self.assertFalse(spmd_overlap._graph_optimization._optimizing)

        # compare overlap vs DDP
        for i, (p1, p2) in enumerate(
            zip(ddp_model.parameters(), spmd_overlap.parameters())
        ):

            assert p1.grad.allclose(  # type: ignore
                p2.grad / self.world_size
            ), "Mismatch in resulting grads between DDP and SPMD."

            self.assertTrue(
                p1.grad.allclose(p2.grad / self.world_size)
                or p1.grad.allclose(p2.grad)
            )

        # compare overlap vs no overlap
        for i, (p1, p2) in enumerate(
            zip(spmd_base.parameters(), spmd_overlap.parameters())
        ):

            self.assertTrue(p1.grad.allclose(p2.grad))

    """@with_comms
    def test_overlap_pass_called(self):
        import torch.fx as fx

        @dataclass
        class RunOverlapProfile:
            num_calls: int

        @contextmanager
        def overlap_profiler() -> Generator[RunOverlapProfile, None, None]:

            original_run_overlap = graph_optimization.run_overlap
            profile: RunOverlapProfile = RunOverlapProfile(num_calls=0)

            # pyre-ignore[53]
            def patched_run_overlap(
                gm: fx.GraphModule,
            ) -> None:
                original_run_overlap(gm)
                profile.num_calls += 1
                print(
                    f"Entered patched run overlap *************************************************"
                )

            try:
                # pyre-ignore[9]
                run_overlap = patched_run_overlap
                print(
                    f" profiler called ************************************************"
                )
                print(f"{profile=}")
                print(f"{run_overlap=}")
                print(f"{patched_run_overlap=}")
                yield profile
            finally:
                run_overlap = original_run_overlap
                print(
                    f"Exiting profiler *********************************************"
                )

        with overlap_profiler() as profiler:
            model_layer_size = 2
            gpu_placement = torch.arange(self.world_size)

            class OverlapModel(nn.Module):
                def __init__(
                    self, layer_count: int = 2, _with_bias: bool = False
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

            spmd_overlap = SPMD(
                base_model,
                schema=Schema(
                    mesh=DeviceMesh(self.device_type, gpu_placement),
                    placements=[Replicate()],
                ),
                optimize_first_iter=True,
            )

            input = torch.randn(2, 10).to(self.device_type)

            spmd_overlap(input).sum().backward()

        self.assertEqual(
            profiler.num_calls, 1, "Expected a single call to run_overlap."
        )
        """


if __name__ == "__main__":
    run_tests()
