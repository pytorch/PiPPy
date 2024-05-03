# Copyright (c) Meta Platforms, Inc. and affiliates
# Minimal effort to run this code:
# $ torchrun --nproc-per-node 3 example_manual_stage.py

import os
import torch
from pippy import ScheduleGPipe, ManualPipelineStage

in_dim = 512
layer_dims = [512, 1024, 256]
out_dim = 10

# Single layer definition
class MyNetworkBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


# Model chunk definition
class ModelChunk0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = MyNetworkBlock(in_dim, layer_dims[0])

    def forward(self, x):
        return self.layer0(x)

class ModelChunk1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MyNetworkBlock(layer_dims[0], layer_dims[1])

    def forward(self, x):
        return self.layer1(x)

class ModelChunk2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = MyNetworkBlock(layer_dims[1], layer_dims[2])
        # Final output layer (with OUT_DIM projection classes)
        self.output_proj = torch.nn.Linear(layer_dims[2], out_dim)

    def forward(self, x):
        x = self.layer2(x)
        return self.output_proj(x)

# To run a distributed training job, we must launch the script in multiple
# different processes. We are using `torchrun` to do so in this example.
# `torchrun` defines two environment variables: `RANK` and `WORLD_SIZE`,
# which represent the index of this process within the set of processes and
# the total number of processes, respectively.
#
# To learn more about `torchrun`, see
# https://pytorch.org/docs/stable/elastic/run.html

torch.manual_seed(0)
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Initialize distributed environment
import torch.distributed as dist

dist.init_process_group(rank=rank, world_size=world_size)

# Figure out device to use
if torch.cuda.is_available():
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
else:
    device = torch.device("cpu")

# Create the model chunks
batch_size = 32
example_input_stage_0 = torch.randn(batch_size, in_dim, device=device)
example_input_stage_1 = torch.randn(batch_size, layer_dims[0], device=device)
example_input_stage_2 = torch.randn(batch_size, layer_dims[1], device=device)
chunks = 4

rank_model_and_input = {
    0: (ModelChunk0(), example_input_stage_0),
    1: (ModelChunk1(), example_input_stage_1),
    2: (ModelChunk2(), example_input_stage_2),
}

# Pipeline stage is our main pipeline runtime. It takes in the pipe object,
# the rank of this process, and the device.
if rank in rank_model_and_input:
    model, example_input = rank_model_and_input[rank]
    stage = ManualPipelineStage(
        model,
        rank,
        world_size,
        device,
        chunks,
        example_input,
    )
    print(f"Rank {rank} initialized")
else:
    raise RuntimeError("Invalid rank")

# Attach to a schedule
schedule = ScheduleGPipe(stage, chunks)

# Input data
x = torch.randn(batch_size, in_dim, device=device)

# Run the pipeline with input `x`. Divide the batch into 4 micro-batches
# and run them in parallel on the pipeline
if rank == 0:
    schedule.step(x)
else:
    output = schedule.step()

print(f"Rank {rank} finished")
