# Copyright (c) Meta Platforms, Inc. and affiliates
# Minimal effort to run this code:
# $ torchrun --nproc-per-node 4 pippy_unroll.py

import os
import torch
import torch.distributed as dist

from pippy.IR import annotate_split_points, Pipe, PipeSplitWrapper
from pippy.PipelineStage import PipelineStage


class IterationBlock(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


class IterativeNetwork(torch.nn.Module):
    def __init__(self, d_hid, num_iters):
        super().__init__()
        self.num_iters = num_iters
        self.iter_block = IterationBlock(d_hid)
        # 10 output classes
        self.output_proj = torch.nn.Linear(d_hid, 10)

    def forward(self, x):
        for i in range(self.num_iters):
            x = self.iter_block(x)
        return self.output_proj(x)


# We are using `torchrun` to run this example with multiple processes.
# `torchrun` defines two environment variables: `RANK` and `WORLD_SIZE`.
torch.manual_seed(0)
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Figure out device to use
if torch.cuda.is_available():
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
else:
    device = torch.device("cpu")

# Create the model
d_hid = 512
# (n-1) iterations + 1 output projection
num_iters = world_size - 1
model = IterativeNetwork(d_hid, num_iters).to(device)

# Add a split point after each iter_block
annotate_split_points(
    model,
    {"iter_block": PipeSplitWrapper.SplitPoint.END},
)

batch_size = 32
example_input = torch.randn(batch_size, d_hid, device=device)
chunks = world_size

pipe = Pipe.from_tracing(model, chunks, example_args=(example_input,))

if rank == 0:
    print(" pipe ".center(80, "*"))
    print(pipe)
    print(" submod0 ".center(80, "*"))
    print(pipe.split_gm.submod_0)

# Initialize distributed environment
dist.init_process_group(rank=rank, world_size=world_size)

# Pipeline stage is our main pipeline runtime. It takes in the pipe object,
# the rank of this process, and the device.
stage = PipelineStage(pipe, rank, device)

# Input data
x = torch.randn(batch_size, d_hid, device=device)

# Run the pipeline with input `x`. Divide the batch into n micro-batches
# and run them in parallel on the pipeline
if rank == 0:
    stage(x)
elif rank == world_size - 1:
    output = stage()
else:
    stage()

if rank == world_size - 1:
    # Run the original code and get the output for comparison
    reference_output = model(x)
    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)
    print(" Pipeline parallel model ran successfully! ".center(80, "*"))
