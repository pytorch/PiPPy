import torch
import torch.fx

class SampleModuleToPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(512, 512)
        self.lin2 = torch.nn.Linear(512, 512)
        self.lin3 = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.lin1(x)
        skip = x
        x = self.lin2(x)
        x = x + skip
        x = self.lin3(x)
        return x

traced = torch.fx.symbolic_trace(SampleModuleToPipeline())

print(traced)