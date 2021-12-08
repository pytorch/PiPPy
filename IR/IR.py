import torch.fx

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