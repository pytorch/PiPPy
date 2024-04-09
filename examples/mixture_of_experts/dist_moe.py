# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# torchrun --nproc-per-node 5 dist_moe.py
# You need use 5 ranks because there are 3 experts, one pre-processor and one gatherer.

"""
                pre-proc
            /       |       \
    expert 0    expert 1   expert 2
            \       |       /
                gatherer
"""

import torch
import torch.distributed as dist

from pippy import annotate_split_points, pipeline, PipelineStage, SplitPoint
from pippy.PipelineSchedule import PipelineScheduleGPipe


d_hid = 16
n_experts = 3
batch_size = 4

torch.manual_seed(0)

# Each expert is a MLP
class ExpertLayer(torch.nn.Module):
    def __init__(self, d_hid) -> None:
        super(ExpertLayer, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x) -> torch.Tensor:
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x

# Full model comprising n experts
class MoE(torch.nn.Module):
    def __init__(self, n_experts: int) -> None:
        super().__init__()
        self.pre_proc = torch.nn.Linear(d_hid, d_hid)
        self.experts = torch.nn.ModuleList(
            [
                ExpertLayer(d_hid)
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_proc(x)
        outputs = []
        for expert in self.experts:
            outputs.append(expert(x))
        return torch.cat(outputs, dim=1)


dist.init_process_group()
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")

model = MoE(n_experts)
x = torch.randn(batch_size, d_hid)

# Mark the split point for each expert
annotate_split_points(model, {f"pre_proc": SplitPoint.END})
for i in range(n_experts):
    annotate_split_points(
        model, {f"experts.{i}": SplitPoint.END}
    )

pippy_model = pipeline(model, 1, (x,))

assert pippy_model.num_stages == world_size
if rank == 0:
    print("Original model:\n", model)
    print("PiPPy model:")
    pippy_model.print_readable()

# Check representation equivalence
ref_out = model(x)
pippy_out = pippy_model(x)[0]
torch.testing.assert_close(pippy_out, ref_out)
print(f"PiPPy model equivalent: {torch.sum(pippy_out)} ref {torch.sum(ref_out)}")

# Create distributed runtime
expert = PipelineStage(pippy_model, rank, device=device)

# Attach to a schedule
# Use a microbatch of 1, i.e. no pipelining
schedule = PipelineScheduleGPipe(expert, 1)
if rank == 0:
    x = x.to(device)
    schedule.step(x)
else:
    dist_out = schedule.step()

# Check equivalence
if rank == dist.get_world_size() - 1:
    print(f"Distributed model equivalent: {torch.sum(dist_out)} ref {torch.sum(ref_out)}")
    print(f"dist_out: {dist_out.shape}")
    print(f"ref_out: {ref_out.shape}")
