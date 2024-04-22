# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from pippy import annotate_split_points, Pipe, SplitPoint


d_hid = 16
n_layers = 8
batch_size = 4


class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super(MLPModule, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class TransformerLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[MLPModule(d_hid) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


transformer = TransformerLike()
print("Original model:\n", transformer)
x = torch.randn(batch_size, d_hid)

# Split into 2 stages
annotate_split_points(
    transformer, {f"layers.{n_layers // 2}": SplitPoint.BEGINNING}
)

pipe = Pipe.from_tracing(
    transformer,
    1,
    (x,),
)
assert pipe.num_stages == 2


def get_layers(module):
    layers = [name for name, _ in module.layers.named_children()]
    return layers


# Collect all layers in pipe
layers = []
for stage_idx in range(pipe.num_stages):
    stage_mod = pipe.get_stage_module(stage_idx)
    print(f"\nStage {stage_idx}: \n", stage_mod)
    layers += get_layers(stage_mod)

# Check layer completeness
orig_layers = get_layers(transformer)
assert sorted(layers) == sorted(orig_layers), f"{layers} != {orig_layers}"
print(f"Layers matched! ", layers)

# Check equivalence
ref = transformer(x)
out = pipe(x)[0]
torch.testing.assert_close(out, ref)
print(f"\nEquivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")
