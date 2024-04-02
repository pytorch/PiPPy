# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from pippy.ManualPipelineStage import ManualPipelineStage
from pippy.PipelineSchedule import PipelineScheduleGPipe
from torch.distributed.device_mesh import init_device_mesh

# torch.testing._internal.common_distributed requies "expecttest"
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA


class DDPAROnce(torch.nn.Module):
    def __init__(
        self, module: torch.nn.Module, group: dist.ProcessGroup, dtype=None
    ):
        super().__init__()
        self.module = module
        self.group = group
        self.dtype = dtype
        # Broadcast the init state of the module from source rank (rank 0)
        global_rank = dist.get_global_rank(self.group, self.group.rank())
        for param in self.module.parameters():
            dist.broadcast(
                param.data,
                src=global_rank,
                group=self.group,
            )
        # Create buffer as 1D tensor
        self.buffer = (
            torch.zeros(
                sum([p.numel() for p in module.parameters()]),
            ).cuda(global_rank)
            if self.dtype is None
            else torch.zeros(
                sum([p.numel() for p in module.parameters()]),
            )
            .to(self.dtype)
            .cuda(global_rank)
        )

    def zero_grad(self):
        self.buffer.zero_()
        offset = 0
        for p in self.module.parameters():
            p.grad = self.buffer[offset : (offset + p.numel())].view(p.shape)
            offset = offset + p.numel()

    def all_reduce_async(self, norm_factor: int):
        self.buffer.div_(norm_factor * self.group.size())
        work = dist.all_reduce(self.buffer, async_op=True, group=self.group)
        return work

    def all_reduce(self, norm_factor: int):
        self.buffer.div_(norm_factor * self.group.size())
        work = dist.all_reduce(self.buffer, async_op=True, group=self.group)
        work.wait()

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


# python -m unittest test_pipeline_with_ddp.TestPipelineDDP.<test>
#               or
# pytest test_pipeline_with_ddp.py -vsk <test>
class TestPipelineDDP(MultiProcessTestCase):
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

    def _create_manual_pipeline_stage(
        self,
        model,
        stage_id,
        num_stages,
        device,
        group,
        inputs,
        num_microbatches,
    ):
        return ManualPipelineStage(
            module=model,
            stage_id=stage_id,
            num_stages=num_stages,
            device=device,
            group=group,
            num_microbatches=num_microbatches,
            input_args=inputs,
        )

    def test_manual_pipeline_with_manual_allreduce(self):
        device = f"cuda:{self.rank}"
        dist.init_process_group(
            init_method=self.init_method,
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )
        # 2 pipeline stages, 2 ddp groups
        #       PP0     PP1
        # DP0    0       2
        #        v       v
        # DP1    1       3
        device_mesh = init_device_mesh(
            device, mesh_shape=(2, 2), mesh_dim_names=("dp", "pp")
        )
        # dp: rows
        # pp: columns
        print(
            f"{self.rank} : {device_mesh} {device_mesh['pp']}, {device_mesh['dp']}"
        )
        pp_group = device_mesh["pp"].get_group()
        ddp_group = device_mesh["dp"].get_group()
        assert type(pp_group) == dist.ProcessGroup
        assert type(ddp_group) == dist.ProcessGroup

        # create "entire model"
        pp_group_size = pp_group.size()

        # 8 layers
        layers_per_model = 4
        full_model = nn.ModuleList(
            [nn.Linear(10, 10) for _ in range(pp_group_size * layers_per_model)]
        )

        # divide the model (8 layers) by the number of ranks (2)
        partial_model = nn.Sequential(
            *full_model[
                pp_group.rank()
                * layers_per_model : (pp_group.rank() + 1)
                * layers_per_model
            ]
        )
        partial_model.to(device)

        # apply "DDP"
        ddp_pp_model = DDPAROnce(partial_model, ddp_group)

        # apply PP
        input1 = torch.rand((1, 10), device=device)
        num_microbatches = 8
        pipeline_stage = self._create_manual_pipeline_stage(
            ddp_pp_model,
            pp_group.rank(),
            pp_group.size(),
            device,
            pp_group,
            input1,
            num_microbatches,
        )

        pipeline_schedule = PipelineScheduleGPipe(
            pipeline_stage,
            n_microbatches=num_microbatches,
        )
        microbatches = [input1.clone() for _ in range(8)]
        pipeline_schedule.step_microbatches(arg_mbs=microbatches)
        print(f"{self.rank} finished pipeline step")

        # all reduce
        ddp_pp_model.all_reduce(num_microbatches)
        print(f"{self.rank} finished all_reduce")


if __name__ == "__main__":
    unittest.main()
