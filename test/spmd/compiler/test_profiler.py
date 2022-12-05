import torch
import torch.nn as nn
from spmd.compiler.api import Schema, SPMD
from spmd.compiler.graph_utils import OP
from spmd.tensor import DeviceMesh, Replicate
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])

    def forward(self, x):
        return sum([m(x) for m in self.module_list])


class TestProfiler(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_basic_profiler(self) -> None:
        def forward_loss(model: nn.Module, *args, **kwargs) -> torch.Tensor:
            return model(*args, **kwargs).sum()

        model = BoringModel().to(self.device_type)
        model = SPMD(
            model,
            schema=Schema(
                mesh=DeviceMesh(
                    self.device_type, torch.arange(self.world_size)
                ),
                placements=[Replicate()],
            ),
            optimize_first_iter=False,
            apply_optimization=True,
        )
        batch = torch.randn(5, 10).to(self.device_type)
        for _ in range(5):
            model(batch).sum().backward()

        for node in model._dist_graph.fwd_graph_modules[0].graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            self.assertTrue(
                node in model._dist_graph.fwd_profilers[0].node_info,
                f"{node.name} {node.op}",
            )

        for node in model._dist_graph.bwd_graph_modules[0].graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            self.assertTrue(
                node in model._dist_graph.bwd_profilers[0].node_info,
                f"{node.name} {node.op}",
            )


if __name__ == "__main__":
    run_tests()
