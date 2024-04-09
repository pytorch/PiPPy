# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn

from pippy import pipeline
from pippy.IR import annotate_split_points, SplitPoint

from pippy.ManualPipelineStage import ManualPipelineStage
from pippy.PipelineSchedule import PipelineScheduleGPipe
from pippy.PipelineStage import PipelineStage

# torch.testing._internal.common_distributed requires "expecttest"
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)

# Example models and helper utils
##########################


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.w1(x)
        x = self.w2(x)
        x = self.relu(x)
        return x


# Tests defined below
##########################


# python -m unittest test_pipeline_stage.TestPipelineStage.<test>
#               or
# pytest test_pipeline_stage.py -vsk <test>
class TestPipelineStage(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        # covers first_stage, middle_stage, last_stage cases
        return 2

    @property
    def init_method(self) -> str:
        return f"{FILE_SCHEMA}{self.file_name}"

    def setUp(self):
        super().setUp()
        # starts world_size processes
        self._spawn_processes()

    def init_distributed(self, use_cuda):
        if use_cuda:
            torch.cuda.set_device(self.rank)
            dist.init_process_group(
                init_method=self.init_method,
                backend="nccl",
                rank=self.rank,
                world_size=self.world_size,
            )
        else:
            dist.init_process_group(
                init_method=self.init_method,
                backend="gloo",
                rank=self.rank,
                world_size=self.world_size,
            )

    @parametrize("pipeline_stage_type", ["manual", "tracing"])
    def test_pipeline_stage(self, pipeline_stage_type):
        # TODO: parameterize
        use_cuda = True

        device = (
            torch.device(f"cuda:{self.rank}")
            if use_cuda
            else torch.device("cpu")
        )
        self.init_distributed(use_cuda=use_cuda)

        in_dim = hidden_dim = out_dim = 10
        model = MLP(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(
            device
        )
        batch_size = 32
        example_input = torch.randn(batch_size, in_dim, device=device)
        chunks = 2

        if pipeline_stage_type == "tracing":
            annotate_split_points(
                model,
                {
                    "w1": SplitPoint.END,
                },
            )
            pipe = pipeline(model, chunks, example_args=(example_input,))
            stage = PipelineStage(pipe, self.rank, device)
        elif pipeline_stage_type == "manual":
            stage = ManualPipelineStage(
                model,
                self.rank,
                self.world_size,
                device,
                chunks,
                input_args=example_input.chunk(chunks)[0],
            )
        else:
            raise ValueError(
                f"Unknown pipeline stage type {pipeline_stage_type}"
            )

        # Define a loss function
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Attach to a schedule
        schedule = PipelineScheduleGPipe(stage, chunks, loss_fn=loss_fn)

        # Input data
        x = torch.randn(batch_size, in_dim, device=device)
        target = torch.randn(batch_size, out_dim, device=device)

        # Run the pipeline with input `x`. Divide the batch into 4 micro-batches
        # and run them in parallel on the pipeline
        if self.rank == 0:
            schedule.step(x)
        elif self.rank == self.world_size - 1:
            losses = []
            output = schedule.step(target=target, losses=losses)
        else:
            schedule.step()


instantiate_parametrized_tests(TestPipelineStage)

if __name__ == "__main__":
    unittest.main()
