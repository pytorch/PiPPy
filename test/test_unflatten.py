# Copyright (c) Meta Platforms, Inc. and affiliates
import pippy
import torch
from pippy import Pipe, pipe_split


# Building block for model
class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        self.lin0 = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(256, 256)

    def forward(self, x: torch.Tensor, constant=None) -> torch.Tensor:
        x = self.conv(x)
        x = self.lin0(x)
        pipe_split()
        x.add_(constant)
        x = self.lin1(x)
        return self.relu(x)


# Full model
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = Block()
        self.block1 = Block()

    def forward(self, x: torch.Tensor, constant=None) -> torch.Tensor:
        x = self.block0(x, constant=constant)
        pipe_split()
        x = self.block1(x, constant=constant)
        return x


x = torch.randn(1, 16, 256, 256)
constant = torch.ones(1, 16, 256, 256)

mod = M()
print("Original model:\n", mod)

pipe = Pipe.from_tracing(
    mod,
    1,
    (x,),
    {"constant": constant},
)

assert pipe.num_stages == 4
gm = pipe.split_gm
orig_state_dict = mod.state_dict()

# Check qualnames
print("\nParameters of each stage:")
for name, submod in gm.named_children():
    print(f"\nStage {name}:")
    for param_name, param in submod.named_parameters():
        assert (
            param_name in orig_state_dict
        ), f"{param_name} not in original state dict"
        print(f"{param_name}: {param.size()}")

# Check equivalence
ref = mod(x, constant)
out = pipe(x, constant)[0]
torch.testing.assert_close(out, ref)
print(f"\nEquivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")
