import torch.fx
from typing import List, NamedTuple

def trace_training_loop(model, loop):
    class TracedMicroBatchTrainingLoop(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            # TODO: generalize to more than one input
            return loop(x)

    traced = torch.fx.symbolic_trace(TracedMicroBatchTrainingLoop())
    return MicroBatchCode(traced)

class MicroBatchCode:
    def __init__(self, traced_training_loop):
        self.traced_training_loop = traced_training_loop

    traced_training_loop : torch.fx.GraphModule

    def __call__(self, *args, **kwargs):
        return self.traced_training_loop(*args, **kwargs)


# Given a microbatch training loop, for example:
#   def loop(self, x):
#     x = self.layer0(x)
#     skip = x
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = x + skip
#     loss = x.sum()
#     x.backward()
#
# We'd like to partition this such that we can conceptually emit
# code like the following:
#
#   def stage0(x):
#     return self.layer0(x)
#
#   def stage1(x):
#     return self.layer1(x)
#
#   def stage2(x);
#     return self.layer2(x)
#
#   def stage3(x, skip):
#     x = x + skip
#     loss = x.sum()
#     return loss
#
#   def loop_pipelined(self, x):
#     x0 = stage0(x)
#     x1 = stage1(x)
#     x2 = stage2(x)
#     loss = stage3(x, x0)
#     loss.backward()
#
#  TODO: multiple uses of modules/parameters
#  TODO: multiple outputs from a stage
#
# By emitting code where stages are invocations and values are
# lexically scoped, we can generate the information each stage
# needs about its data dependencies. We can encode this information
# as types and construct type expressions for each argument on each
# stage.