# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import random
import time
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn

from pippy import Schedule1F1B, ScheduleInterleaved1F1B

from pippy.ManualPipelineStage import (
    create_metadata_tensor,
    extract_metadata_from_tensor,
    get_stage_shapes,
    ManualPipelineStage,
    validate_stage_shapes,
)

# torch.testing._internal.common_distributed requies "expecttest"
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import FILE_SCHEMA

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


class ModelWithSleep(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int,
        rank: int,
    ):
        super().__init__()
        self.in_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()
        self.rank = rank

    def forward(self, x):
        x = self.in_layer(x)
        x = self.middle(x)
        # this delay helps to simulate inconsistencies in timing between ranks
        if self.rank == 0 or self.rank == 1:
            time.sleep(random.uniform(0, 0.5))
        x = self.out_layer(x)
        x = self.relu(x)
        return x


# Tests defined below
##########################


# python -m unittest test_pipeline_schedule.TestPipelineSchedule.<test>
#               or
# pytest test_pipeline_schedule.py -vsk <test>
class TestPipelineSchedule(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        # covers first_stage, middle_stage, last_stage cases
        return 4

    @property
    def init_method(self) -> str:
        return f"{FILE_SCHEMA}{self.file_name}"

    def setUp(self):
        super().setUp()
        # starts world_size processes
        self._spawn_processes()

    def init_distributed(self):
        dist.init_process_group(
            init_method=self.init_method,
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
        )

    def _create_pipeline_stage(
        self,
        model,
        inputs,
        device,
        num_microbatches=None,
        stage_id=None,
        num_stages=None,
    ):
        return ManualPipelineStage(
            module=model,
            stage_id=self.rank if stage_id is None else stage_id,
            num_stages=self.world_size if num_stages is None else num_stages,
            input_args=inputs,
            device=device,
            num_microbatches=num_microbatches,
        )

    def _create_virtual_pipeline_stages(
        self, model, inputs, device, virtual_size, num_microbatches=None
    ):
        virtual_stages = []
        for i in range(virtual_size):
            stage = self._create_pipeline_stage(
                model,
                inputs,
                device,
                num_microbatches,
                self.rank + (i * self.world_size),
                self.world_size * virtual_size,
            )
            virtual_stages.append(stage)
        return virtual_stages

    def test_pipeline_stage_init(self):
        # TODO: parameterize the device?
        device = "cpu"
        self.init_distributed()

        model = MLP(dim=8, hidden_dim=4, out_dim=4)
        inputs = torch.rand((2, 8), device=device)
        self._create_pipeline_stage(model, inputs, device, 1)
        with self.assertRaises(TypeError):
            invalid_input_args = {"foo": "bar"}
            self._create_pipeline_stage(model, invalid_input_args, device, 1)

        with self.assertRaises(TypeError):
            invalid_model = InvalidOutputModel()
            self._create_pipeline_stage(invalid_model, inputs, device, 1)

    def test_pipeline_stage_fwd(self):
        # TODO: parameterize the device?
        device = "cpu"
        self.init_distributed()

        # single input model forward
        model = MLP(dim=8, hidden_dim=4, out_dim=4)
        input1 = torch.rand((2, 8), device=device)
        pipeline_stage = self._create_pipeline_stage(model, input1, device, 1)
        output = pipeline_stage.forward_one_chunk([input1])
        self.assertEqual(output.shape, torch.Size([2, 4]))

        # multi-input model forward
        model = MultiInputArgMLP(dim1=8, dim2=4, out_dim=4)
        input1 = torch.rand((2, 8), device=device)
        input2 = torch.rand((2, 4), device=device)
        pipeline_stage = self._create_pipeline_stage(
            model, [input1, input2], device, 1
        )
        output = pipeline_stage.forward_one_chunk([input1, input2])
        if self.rank == self.world_size - 1:
            self.assertIsInstance(output, torch.Tensor)
        else:
            self.assertEqual(output.shape, torch.Size([2, 4]))

        # multi-output model forward
        model = MultiOutputArgMLP(dim=8, out_dim=4)
        input1 = torch.rand((2, 8), device=device)
        pipeline_stage = self._create_pipeline_stage(model, input1, device, 1)
        output = pipeline_stage.forward_one_chunk([input1])
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, torch.Size([2, 4]))
        self.assertEqual(output[1].shape, torch.Size([4, 4]))

    def test_get_stage_shapes(self):
        device = "cpu"
        self.init_distributed()

        model_chunk = MLP(dim=8, hidden_dim=4, out_dim=8)
        microbatch = torch.rand((10, 8), device=device)
        # test single model
        stages_shapes = get_stage_shapes(
            stage_modules=[model_chunk],
            stage_ids=[self.rank],
            num_stages=self.world_size,
            rank=self.rank,
            world_size=self.world_size,
            device=device,
            microbatch=microbatch if self.rank == 0 else None,
        )
        self.assertEqual(len(stages_shapes), 1)
        shapes = stages_shapes[self.rank]
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes["inputs"], [torch.Size([10, 8])])
        self.assertEqual(shapes["outputs"], [torch.Size([10, 8])])

        # test multiple models (multiple stages)
        model_chunk1 = MLP(dim=8, hidden_dim=4, out_dim=8)
        model_chunk2 = MLP(dim=8, hidden_dim=4, out_dim=8)
        stages_shapes = get_stage_shapes(
            stage_modules=[model_chunk1, model_chunk2],
            stage_ids=[self.rank, self.rank + self.world_size],
            num_stages=self.world_size * 2,
            rank=self.rank,
            world_size=self.world_size,
            device=device,
            microbatch=microbatch if self.rank == 0 else None,
        )
        self.assertEqual(len(stages_shapes), 2)
        self.assertEqual(len(stages_shapes[self.rank]), 2)
        shapes = stages_shapes[self.rank + self.world_size]
        self.assertEqual(shapes["inputs"], [torch.Size([10, 8])])
        self.assertEqual(shapes["outputs"], [torch.Size([10, 8])])

    def test_validate_stage_shapes(self):
        device = "cpu"
        self.init_distributed()

        # test single pipeline stage
        model_chunk = MLP(dim=8, hidden_dim=4, out_dim=8)
        input1 = torch.rand((4, 8), device=device)
        pipeline_stage = self._create_pipeline_stage(
            model_chunk, input1, device, 1
        )
        validate_stage_shapes([pipeline_stage])

        # test multiple pipeline stages
        model_chunk = MLP(dim=2, hidden_dim=2, out_dim=2)
        input1 = torch.rand((5, 2), device=device)
        stages = self._create_virtual_pipeline_stages(
            model_chunk, input1, device, 2, 1
        )
        validate_stage_shapes(stages)

        # mismatched model chunk case
        with self.assertRaises(ValueError):
            if self.rank == 1:
                model_chunk = MLP(dim=2, hidden_dim=4, out_dim=6)
            pipeline_stage = self._create_pipeline_stage(
                model_chunk, input1, device, 1
            )
            validate_stage_shapes([pipeline_stage])

    @skip_if_lt_x_gpu(4)
    def test_1f1b(self):
        device = torch.device(f"cuda:{self.rank}")
        dist.init_process_group(
            init_method=self.init_method,
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )

        # test single pipeline stage
        model = MLP(dim=8, hidden_dim=4, out_dim=8)
        microbatch = torch.rand((4, 8), device=device)
        stage = self._create_pipeline_stage(model, microbatch, device, 8)
        num_microbatches = 8
        microbatches = [
            [torch.randn_like(microbatch)] for _ in range(num_microbatches)
        ]

        schedule = Schedule1F1B(stage, num_microbatches)
        schedule._step_microbatches(microbatches)
        dist.barrier()

    @skip_if_lt_x_gpu(4)
    def test_interleaved_1f1b(self):
        # TODO: not working
        return

        device = torch.device(f"cuda:{self.rank}")
        dist.init_process_group(
            init_method=self.init_method,
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )

        # num local pipeline stages < world_size
        model = MLP(dim=8, hidden_dim=4, out_dim=8)
        microbatch = torch.rand((4, 8), device=device)
        num_microbatches = 8
        stages = self._create_virtual_pipeline_stages(
            model, microbatch, device, 2, num_microbatches=num_microbatches
        )
        microbatches = [
            (torch.randn_like(microbatch),) for _ in range(num_microbatches)
        ]

        schedule = ScheduleInterleaved1F1B(
            stages,
            num_microbatches,
        )
        schedule._step_microbatches(microbatches)

        # num local pipeline stages == world_size
        num_microbatches = 8
        stages = self._create_virtual_pipeline_stages(
            model,
            microbatch,
            device,
            self.world_size,
            num_microbatches=num_microbatches,
        )
        microbatches = [
            torch.randn_like(microbatch) for _ in range(num_microbatches)
        ]

        schedule = ScheduleInterleaved1F1B(
            stages,
            num_microbatches,
        )
        schedule._step_microbatches(microbatches)

        # differing microbatch size
        num_microbatches = 64
        microbatches = [
            torch.randn_like(microbatch) for _ in range(num_microbatches)
        ]
        schedule._step_microbatches(microbatches)

    def test_interleaved_1f1b_negative(self):
        device = torch.device("cpu")
        dist.init_process_group(
            init_method=self.init_method,
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
        )

        model = MLP(dim=8, hidden_dim=4, out_dim=8)
        microbatch = torch.rand((4, 8))
        num_microbatches = 4

        # requires at least two stages
        with self.assertRaises(ValueError):
            stages = self._create_virtual_pipeline_stages(
                model, microbatch, device, 1, num_microbatches=num_microbatches
            )
            schedule = ScheduleInterleaved1F1B(
                stages,
                num_microbatches,
            )

        stages = self._create_virtual_pipeline_stages(
            model, microbatch, device, 4, num_microbatches=num_microbatches
        )
        schedule = ScheduleInterleaved1F1B(
            stages,
            num_microbatches,
        )

        # invalid microbatch values
        with self.assertRaises(ValueError):
            num_microbatches = 1
            microbatches = [
                torch.randn_like(microbatch) for _ in range(num_microbatches)
            ]
            schedule._step_microbatches(microbatches)

        # invalid microbatch values
        with self.assertRaises(ValueError):
            num_microbatches = 5
            microbatches = [
                torch.randn_like(microbatch) for _ in range(num_microbatches)
            ]
            schedule._step_microbatches(microbatches)

    @skip_if_lt_x_gpu(4)
    def test_interleaved_1f1b_with_model_sleep(self):
        device = torch.device(f"cuda:{self.rank}")
        dist.init_process_group(
            init_method=self.init_method,
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )

        num_dims = 4
        model = ModelWithSleep(
            dim=num_dims, hidden_dim=8, out_dim=num_dims, rank=self.rank
        )
        stages_per_rank = 2
        num_microbatches_list = [4, 8, 16]
        for num_microbatches in num_microbatches_list:
            batch = torch.rand((num_microbatches, num_dims), device=device)
            stages = self._create_virtual_pipeline_stages(
                model,
                torch.rand((1, num_dims)).to("meta"),
                device,
                stages_per_rank,
                num_microbatches=num_microbatches,
            )

            schedule = ScheduleInterleaved1F1B(
                stages, num_microbatches, loss_fn=nn.MSELoss()
            )
            if self.rank == 0:
                schedule.step(batch)
            elif self.rank == self.world_size - 1:
                target = torch.rand((num_microbatches, num_dims), device=device)
                losses = []
                schedule.step(target=target, losses=losses)
            else:
                schedule.step()
            dist.barrier()
            torch.cuda.synchronize()
            print(f"Finished with testing {num_microbatches} microbatches")

    def test_check_inputs(self):
        device = (
            torch.device(f"cuda:{self.rank}")
            if torch.cuda.is_available()
            else "cpu"
        )

        dist.init_process_group(
            init_method=self.init_method,
            backend="nccl" if dist.is_nccl_available() else "gloo",
            rank=self.rank,
            world_size=self.world_size,
        )

        # test single pipeline stage
        model = MLP(dim=8, hidden_dim=4, out_dim=8)
        microbatch = torch.rand((4, 8), device=device)
        num_microbatches = 8
        stage = self._create_pipeline_stage(
            model, microbatch, device, num_microbatches
        )

        schedule = Schedule1F1B(stage, num_microbatches)
        # invalid input length
        with self.assertRaises(ValueError):
            invalid_microbatches = [(i,) for i in range(7)]
            schedule._step_microbatches(invalid_microbatches)

        # invalid input shapes
        with self.assertRaises(ValueError):
            invalid_microbatches = [(torch.ones(8, 4, 8))]
            schedule._step_microbatches(invalid_microbatches)

        # invalid input type
        with self.assertRaises(TypeError):
            invalid_microbatches = torch.ones(8, 4, 8)
            schedule._step_microbatches(invalid_microbatches)

        # invalid loss
        with self.assertRaises(TypeError):
            loss = 1
            microbatches = [
                torch.randn_like(microbatch) for _ in range(num_microbatches)
            ]
            schedule._step_microbatches(microbatches, loss=loss)


class UtilTest(unittest.TestCase):
    def test_metadata_tensor(self):
        # scalar
        t1 = torch.tensor(1)
        # 1d (3)
        t2 = torch.tensor([1, 2, 3])
        # 2d (2x3)
        t3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        # n-d
        t4 = torch.ones((3, 4, 5, 6))

        metadata_tensor = create_metadata_tensor([t1, t2, t3, t4])
        shapes = extract_metadata_from_tensor(metadata_tensor)

        self.assertEqual(shapes[0], t1.shape)
        self.assertEqual(shapes[1], t2.shape)
        self.assertEqual(shapes[2], t3.shape)
        self.assertEqual(shapes[3], t4.shape)


if __name__ == "__main__":
    unittest.main()
