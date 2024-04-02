# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from pippy import annotate_split_points, Pipe, SplitPoint
import torch.distributed.checkpoint as dcp
import tempfile


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
            *[
                # torch.nn.TransformerEncoderLayer(
                #     d_model=16,
                #     nhead=4,
                #     dim_feedforward=16,
                #     batch_first=True,
                #     activation="gelu",
                #     dropout=0.0,
                # )
                MLPModule(d_hid)
                for _ in range(n_layers)
            ]
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


def pipe_to_sd(pipe):
    sd = {}
    for stage_idx in range(pipe.num_stages):
        stage_mod = pipe.get_stage_module(stage_idx)
        sd[f"stage_{stage_idx}"] = stage_mod
    return sd

with tempfile.TemporaryDirectory() as tmpdir:
    #Simulate saving the pipe
    # Option 1:
    # for stage_idx in range(pipe.num_stages):
    #     print(f"Saving pipeline stage {stage_idx}")
    #     stage_mod = pipe.get_stage_module(stage_idx)
    #     dcp.save(
    #         {f"stage_{stage_idx}": stage_mod},
    #         checkpoint_id=f"{tmpdir}_{stage_idx}"
    #     )
    # Option 2:
    sd = pipe_to_sd(pipe)
    dcp.save(state_dict, checkpoint_id=tmpdir)


    #Simulate loading the pipe
    # Option 1:
    # for stage_idx in range(pipe.num_stages):
    #     print(f"Loading pipeline stage {stage_idx}")
    #     stage_mod = pipe.get_stage_module(stage_idx)
    #     dcp.load(
    #         {f"stage_{stage_idx}": stage_mod},
    #         checkpoint_id=f"{tmpdir}_{stage_idx}"
    #     )

    #Option 2:
    new_pipe = Pipe.from_tracing(
        transformer,
        1,
        (x,),
    )
    sd = pipe_to_sd(new_pipe)
    dcp.load(sd, checkpoint_id=tmpdir)

pipe = new_pipe

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
