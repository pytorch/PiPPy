# Copyright (c) Meta Platforms, Inc. and affiliates
# Minimal effort to run this code:
# $ torchrun --nproc-per-node 3 example_train.py

import os
import torch
from pippy.IR import annotate_split_points, SplitPoint
from pippy.PipelineSchedule import ScheduleGPipe
from pippy.PipelineStage import PipelineStage

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


# Full model definition
class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = len(layer_dims)

        prev_dim = in_dim
        # Add layers one by one
        for i, dim in enumerate(layer_dims):
            super().add_module(f"layer{i}", MyNetworkBlock(prev_dim, dim))
            prev_dim = dim

        # Final output layer (with OUT_DIM projection classes)
        self.output_proj = torch.nn.Linear(layer_dims[-1], out_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i}")
            x = layer(x)

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

# Figure out device to use
if torch.cuda.is_available():
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
else:
    device = torch.device("cpu")

# Create the model
mn = MyNetwork().to(device)

annotate_split_points(
    mn,
    {
        "layer0": SplitPoint.END,
        "layer1": SplitPoint.END,
    },
)

batch_size = 32
example_input = torch.randn(batch_size, in_dim, device=device)
chunks = 4

from pippy import pipeline
pipe = pipeline(mn, chunks, example_args=(example_input,))

if rank == 0:
    print(" pipe ".center(80, "*"))
    print(pipe)
    print(" stage 0 ".center(80, "*"))
    print(pipe.split_gm.submod_0)
    print(" stage 1 ".center(80, "*"))
    print(pipe.split_gm.submod_1)
    print(" stage 2 ".center(80, "*"))
    print(pipe.split_gm.submod_2)


# Initialize distributed environment
import torch.distributed as dist

dist.init_process_group(rank=rank, world_size=world_size)

# Pipeline stage is our main pipeline runtime. It takes in the pipe object,
# the rank of this process, and the device.
stage = PipelineStage(pipe, rank, device)

# Define a loss function
loss_fn=torch.nn.MSELoss(reduction="sum")

# Attach to a schedule
schedule = ScheduleGPipe(stage, chunks, loss_fn=loss_fn)

# Input data
x = torch.randn(batch_size, in_dim, device=device)
target = torch.randn(batch_size, out_dim, device=device)

# Run the pipeline with input `x`. Divide the batch into 4 micro-batches
# and run them in parallel on the pipeline
if rank == 0:
    schedule.step(x)
elif rank == world_size - 1:
    losses = []
    output = schedule.step(target=target, losses=losses)
else:
    schedule.step()

if rank == world_size - 1:
    # Run the original code and get the output for comparison
    reference_output = mn(x)
    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)
    print(f"Loss of microbatches: {losses}")
    print(" Pipeline parallel model ran successfully! ".center(80, "*"))
