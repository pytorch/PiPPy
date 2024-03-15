# Copyright (c) Meta Platforms, Inc. and affiliates
# Minimal effort to run this code:
# $ torchrun --nproc-per-node 3 example.py

import os
import torch
from pippy.IR import annotate_split_points, Pipe, PipeSplitWrapper
from pippy.PipelineSchedule import PipelineScheduleGPipe
from pippy.PipelineStage import PipelineStage


class MyNetworkBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


class MyNetwork(torch.nn.Module):
    def __init__(self, in_dim, layer_dims):
        super().__init__()

        prev_dim = in_dim
        for i, dim in enumerate(layer_dims):
            setattr(self, f"layer{i}", MyNetworkBlock(prev_dim, dim))
            prev_dim = dim

        self.num_layers = len(layer_dims)
        # 10 output classes
        self.output_proj = torch.nn.Linear(layer_dims[-1], 10)

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f"layer{i}")(x)

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
in_dim = 512
layer_dims = [512, 1024, 256]
mn = MyNetwork(in_dim, layer_dims).to(device)

annotate_split_points(
    mn,
    {
        "layer0": PipeSplitWrapper.SplitPoint.END,
        "layer1": PipeSplitWrapper.SplitPoint.END,
    },
)

batch_size = 32
example_input = torch.randn(batch_size, in_dim, device=device)
chunks = 4

pipe = Pipe.from_tracing(mn, chunks, example_args=(example_input,))

print(" pipe ".center(80, "*"))
print(pipe)
print(" submod0 ".center(80, "*"))
print(pipe.split_gm.submod_0)
print(" submod1 ".center(80, "*"))
print(pipe.split_gm.submod_1)
print(" submod2 ".center(80, "*"))
print(pipe.split_gm.submod_2)


# Initialize distributed environment
import torch.distributed as dist

dist.init_process_group(rank=rank, world_size=world_size)

# Pipeline stage is our main pipeline runtime. It takes in the pipe object,
# the rank of this process, and the device.
stage = PipelineStage(pipe, rank, device)

# Attach to a schedule
schedule = PipelineScheduleGPipe(stage, chunks)

# Input data
x = torch.randn(batch_size, in_dim, device=device)

# Run the pipeline with input `x`. Divide the batch into 4 micro-batches
# and run them in parallel on the pipeline
if rank == 0:
    schedule.step(x)
else:
    output = schedule.step()

if rank == world_size - 1:
    # Run the original code and get the output for comparison
    reference_output = mn(x)
    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)
    print(" Pipeline parallel model ran successfully! ".center(80, "*"))
