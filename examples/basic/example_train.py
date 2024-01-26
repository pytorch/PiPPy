# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from typing import Any


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


mn = MyNetwork(512, [512, 1024, 256])

from pippy.IR import Pipe

pipe = Pipe.from_tracing(mn)
print(pipe)
print(pipe.split_gm.submod_0)


from pippy.IR import annotate_split_points, PipeSplitWrapper

annotate_split_points(
    mn,
    {
        "layer0": PipeSplitWrapper.SplitPoint.END,
        "layer1": PipeSplitWrapper.SplitPoint.END,
    },
)

pipe = Pipe.from_tracing(mn)
print(" pipe ".center(80, "*"))
print(pipe)
print(" submod0 ".center(80, "*"))
print(pipe.split_gm.submod_0)
print(" submod1 ".center(80, "*"))
print(pipe.split_gm.submod_1)
print(" submod2 ".center(80, "*"))
print(pipe.split_gm.submod_2)


# To run a distributed training job, we must launch the script in multiple
# different processes. We are using `torchrun` to do so in this example.
# `torchrun` defines two environment variables: `LOCAL_RANK` and `WORLD_SIZE`,
# which represent the index of this process within the set of processes and
# the total number of processes, respectively.
#
# To learn more about `torchrun`, see
# https://pytorch.org/docs/stable/elastic/run.html
import os

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# PiPPy uses the PyTorch RPC interface. To use RPC, we must call `init_rpc`
# and inform the RPC framework of this process's rank and the total world
# size. We can directly pass values `torchrun` provided.`
#
# To learn more about the PyTorch RPC framework, see
# https://pytorch.org/docs/stable/rpc.html
import torch.distributed.rpc as rpc

rpc.init_rpc(f"worker{local_rank}", rank=local_rank, world_size=world_size)

# PiPPy relies on the concept of a "driver" process. The driver process
# should be a single process within the RPC group that instantiates the
# PipelineDriver and issues commands on that object. The other processes
# in the RPC group will receive commands from this process and execute
# the pipeline stages
if local_rank == 0:
    from pippy.PipelineDriver import PipelineDriverFillDrain
    from pippy.microbatch import TensorChunkSpec

    # LossWrapper is a convenient base class you can use to compose your model
    # with the desired loss function for the purpose of pipeline parallel training.
    # Since the loss is executed as part of the pipeline, it cannot reside in the
    # training loop, so you must embed it like this
    from pippy.IR import LossWrapper

    class ModelLossWrapper(LossWrapper):
        def forward(self, x, target):
            return self.loss_fn(self.module(x), target)

    # TODO: mean reduction
    loss_wrapper = ModelLossWrapper(
        module=mn, loss_fn=torch.nn.MSELoss(reduction="sum")
    )

    # Instantiate the `Pipe` similarly to before, but with two differences:
    #   1) We pass in the `loss_wrapper` module to include the loss in the
    #      computation
    #   2) We specify `output_loss_value_spec`. This is a data structure
    #      that should mimic the structure of the output of LossWrapper
    #      and has a True value in the position where the loss value will
    #      be. Since LossWrapper returns just the loss, we just pass True
    pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=True)

    # We now have two args: `x` and `target`, so specify batch dimension
    # for both.
    args_chunk_spec: Any = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec: Any = {}
    # The output of our model is now a `loss` value, which is a scalar tensor.
    # PiPPy's default is to concatenate outputs, but that will not
    # work with a scalar tensor. So we use a LossReducer instead
    # to merge together the loss values from each microbatch into a
    # single unified loss.
    from pippy.microbatch import LossReducer

    output_chunk_spec: Any = LossReducer(0.0, lambda a, b: a + b)

    # Instantiate the driver as usual.
    driver = PipelineDriverFillDrain(
        pipe,
        64,
        world_size=world_size,
        args_chunk_spec=args_chunk_spec,
        kwargs_chunk_spec=kwargs_chunk_spec,
        output_chunk_spec=output_chunk_spec,
    )

    # Instantiate remote Adam optimizers. `instantiate_optimizer` takes the
    # optimizer class as the first argument, then additional arguments to that
    # optimizer. Note that the `parameters` argument is omitted; PiPPy will
    # populate that value for each pipeline stage for you.
    optimizer = driver.instantiate_optimizer(torch.optim.Adam)
    # Also instantiate a learning rate scheduler. Note that the `optimizer` argument is
    # omitted; PiPPy will populate that argument for each pipeline stage
    lr_scheduler = driver.instantiate_lr_scheduler(
        torch.optim.lr_scheduler.LinearLR, total_iters=100
    )

    N_TRAINING_STEPS = 100

    x = torch.randn(512, 512)
    target = torch.randn(512, 10)
    for i in range(N_TRAINING_STEPS):
        optimizer.zero_grad()
        pipe_loss = driver(x, target)
        optimizer.step()
        lr_scheduler.step()

        log_info = f" Training step {i}, loss: {pipe_loss}, LR: {lr_scheduler.get_last_lr()} "
        print(log_info.center(80, "*"))

    print(" Pipeline parallel model ran successfully! ".center(80, "*"))

rpc.shutdown()
