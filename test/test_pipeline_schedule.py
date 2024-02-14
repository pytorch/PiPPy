# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import torch
import torch.nn as nn
from pippy.PipelineSchedule import PipelineStageV2Impl

# torch.testing._internal.common_distributed requies "expecttest"
from torch.testing._internal.common_distributed import MultiProcessTestCase

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


class MultiInputArgMLP(nn.Module):
    def __init__(
        self,
        dim1: int,
        dim2: int,
        out_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim1, out_dim, bias=False)
        self.w2 = nn.Linear(dim2, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.w1(x)
        y = self.w2(y)
        z = x + y
        z = self.relu(z)
        return z


class MultiOutputArgMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x):
        x = self.w1(x)
        y = torch.cat([x, x], dim=0)
        return x, y


class InvalidOutputModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return {}


# Tests defined below
##########################


# python -m unittest test_pipeline_schedule.TestPipelineSchedule.<test>
#               or
# pytest test_pipeline_schedule.py -vsk <test>
class TestPipelineSchedule(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        # covers first_stage, middle_stage, last_stage cases
        return 3

    def setUp(self):
        super().setUp()
        # starts world_size processes
        self._spawn_processes()

    def _create_pipline_stage(self, model, inputs, device):
        return PipelineStageV2Impl(
            module=model,
            stage_id=self.rank,
            num_stages=self.world_size,
            rank=self.rank,
            world_size=self.world_size,
            input_args=inputs,
            device=device,
        )

    def test_pipeline_stage_init(self):
        # TODO: parameterize the device?
        device = "cpu"

        model = MLP(dim=8, hidden_dim=4, out_dim=4)
        inputs = torch.rand((2, 8), device=device)
        self._create_pipline_stage(model, inputs, device)
        with self.assertRaises(ValueError):
            invalid_input_args = {"foo": "bar"}
            self._create_pipline_stage(model, invalid_input_args, device)

        with self.assertRaises(ValueError):
            invalid_model = InvalidOutputModel()
            self._create_pipline_stage(invalid_model, inputs, device)

    def test_pipeline_stage_fwd(self):
        # TODO: parameterize the device?
        device = "cpu"

        # single input model forward
        model = MLP(dim=8, hidden_dim=4, out_dim=4)
        input1 = torch.rand((2, 8), device=device)
        pipeline_stage = self._create_pipline_stage(model, input1, device)
        output = pipeline_stage(input1)
        self.assertEqual(output.shape, torch.Size([2, 4]))

        # multi-input model forward
        model = MultiInputArgMLP(dim1=8, dim2=4, out_dim=4)
        input1 = torch.rand((2, 8), device=device)
        input2 = torch.rand((2, 4), device=device)
        pipeline_stage = self._create_pipline_stage(
            model, [input1, input2], device
        )
        output = pipeline_stage([input1, input2])
        self.assertEqual(output.shape, torch.Size([2, 4]))

        # multi-output model forward
        model = MultiOutputArgMLP(dim=8, out_dim=4)
        input1 = torch.rand((2, 8), device=device)
        pipeline_stage = self._create_pipline_stage(model, input1, device)
        output = pipeline_stage(input1)
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, torch.Size([2, 4]))
        self.assertEqual(output[1].shape, torch.Size([4, 4]))


if __name__ == "__main__":
    unittest.main()
