# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import copy
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from pippy.ManualPipelineStage import ManualPipelineStage
from pippy.PipelineSchedule import (
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleLoopedBFS,
)
from torch.distributed._composable.fsdp.fully_shard import (
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# torch.testing._internal.common_distributed requies "expecttest"
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)


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


# python -m unittest test_composability.TestPipelineComposability.<test>
#               or
# pytest test_composability.py -vsk <test>
class TestPipelineComposability(MultiProcessTestCase):
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

    def _init_device_mesh(self, mesh_shape, mesh_dim_names):
        device = f"cuda:{self.rank}"
        torch.cuda.set_device(device)
        dist.init_process_group(
            init_method=self.init_method,
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )

        device_mesh = init_device_mesh(
            "cuda", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
        )
        return device_mesh, device

    def test_manual_pipeline_with_manual_allreduce(self):
        # 2 pipeline stages, 2 ddp groups
        #       PP0     PP1
        # DP0    0       2
        #        v       v
        # DP1    1       3
        device_mesh, device = self._init_device_mesh(
            mesh_shape=(2, 2), mesh_dim_names=("dp", "pp")
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

        pipeline_schedule = ScheduleGPipe(
            pipeline_stage,
            n_microbatches=num_microbatches,
        )
        microbatches = [input1.clone() for _ in range(8)]
        pipeline_schedule.step_microbatches(arg_mbs=microbatches)
        print(f"{self.rank} finished pipeline step")

        # all reduce
        ddp_pp_model.all_reduce(num_microbatches)
        print(f"{self.rank} finished all_reduce")

    @parametrize(
        "schedule_name", ["gpipe", "1f1b", "looped_bfs", "interleaved_1f1b"]
    )
    def test_manual_pipeline_with_fsdp(self, schedule_name):
        device_mesh, device = self._init_device_mesh(
            mesh_shape=(2, 2), mesh_dim_names=("dp", "pp")
        )
        pp_group = device_mesh["pp"].get_group()
        dp_mesh = device_mesh["dp"]
        assert type(pp_group) == dist.ProcessGroup
        assert type(dp_mesh) == DeviceMesh

        # create "entire model"
        total_layers = 8
        dim = 10
        full_model = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(total_layers)]
        )
        ref_model = nn.Sequential(*copy.deepcopy(full_model))
        ref_model.to(device)

        def build_stage(stage_idx, num_stages):
            layers_per_model = total_layers // num_stages
            assert layers_per_model * num_stages == total_layers
            # return offset so validation code can match partial layer back to orig model
            offset = stage_idx * layers_per_model
            partial_model = nn.Sequential(
                *full_model[offset : (stage_idx + 1) * layers_per_model]
            )
            partial_model.to(device)

            # apply FSDP
            mp_policy = MixedPrecisionPolicy(
                # TODO(whc) need to fix PP + FSDP-mixed-precision
                # tracer for PP assumes f32 and is caught off guard when runtime FSDP interacts using bf16 inputs
                # param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            )
            fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
            for layer in partial_model.children():
                fully_shard(
                    layer,
                    **fsdp_config,
                    reshard_after_forward=False,
                )
            fsdp_model = fully_shard(partial_model, **fsdp_config)

            stage = self._create_manual_pipeline_stage(
                fsdp_model,
                stage_idx,
                num_stages,
                device,
                pp_group,
                input_mb[0],
                num_microbatches,
            )
            return stage, offset

        # apply PP
        num_microbatches = 8
        inputs = [
            torch.rand((num_microbatches, dim), device=device)
            for _ in range(dp_mesh.size())
        ]
        input = inputs[dp_mesh.get_local_rank()]
        input_mb = [
            [input[i].reshape((1, dim))] for i in range(num_microbatches)
        ]
        # dummy loss needed just to force backwards to run in schedule step
        loss_fn = lambda y, t: y.sum()

        # divide the model (8 layers) by the number of ranks (2)
        if schedule_name in {"looped_bfs", "interleaved_1f1b"}:
            n_virtual = 2
            num_stages = pp_group.size() * n_virtual
            stages = []
            offsets = []
            for i in range(n_virtual):
                stage, offset = build_stage(
                    pp_group.rank() + n_virtual * i, num_stages
                )
                stages.append(stage)
                offsets.append(offset)
                partial_models = [
                    pipeline_stage.submod for pipeline_stage in stages
                ]

            if schedule_name == "looped_bfs":
                pipeline_schedule = ScheduleLoopedBFS(
                    stages,
                    n_microbatches=num_microbatches,
                    loss_fn=loss_fn,
                )
            elif schedule_name == "interleaved_1f1b":
                pipeline_schedule = ScheduleInterleaved1F1B(
                    stages,
                    n_microbatches=num_microbatches,
                    loss_fn=loss_fn,
                )
            else:
                raise RuntimeError(f"unsupported schedule {schedule_name}")
        else:
            pipeline_stage, offset = build_stage(
                pp_group.rank(), pp_group.size()
            )
            partial_models = [pipeline_stage.submod]
            offsets = [offset]

            if schedule_name == "gpipe":
                pipeline_schedule = ScheduleGPipe(
                    pipeline_stage,
                    n_microbatches=num_microbatches,
                    loss_fn=loss_fn,
                )
            elif schedule_name == "1f1b":
                pipeline_schedule = Schedule1F1B(
                    pipeline_stage,
                    n_microbatches=num_microbatches,
                    loss_fn=loss_fn,
                )
            else:
                raise RuntimeError(f"unsupported schedule {schedule_name}")

        pipeline_schedule.step_microbatches(
            arg_mbs=input_mb, target_mbs=input_mb
        )

        # Ref model runs on 2 different inputs, accumulating grads across them.
        # this ensures that we detect if the FSDP reduce becomes a no-op.
        # (in fsdp case, we use one of these inputs on each DP rank)
        (ref_model(inputs[0]).sum()).backward()
        (ref_model(inputs[1]).sum()).backward()

        # simulate the built-in averaging done by FSDP
        for p in ref_model.parameters():
            p.grad /= dp_mesh.size()

        # Validate that whichever weights we have locally match that part of our local/full ref model
        # (we force FSDP's grads to be all-gathered (.full_tensor) to make it simpler)
        ref_parameters = dict(ref_model.named_parameters())
        for partial_model, offset in zip(partial_models, offsets):
            for name, p in partial_model.named_parameters():
                parts = name.split(".")
                parts[0] = str(int(parts[0]) + offset)
                name = ".".join(parts)
                ref_p = ref_parameters[name]
                self.assertTrue(isinstance(p.grad, DTensor))
                self.assertEqual(ref_p.grad, p.grad.full_tensor())


instantiate_parametrized_tests(TestPipelineComposability)

if __name__ == "__main__":
    unittest.main()
