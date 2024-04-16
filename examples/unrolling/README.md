## What does this example do?

This is a synthetic example used to demonstrate PiPPy's functionality in unrolling iterative blocks in a model.

We create a model that runs an iteration block in a for loop:
```python
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
```

If we annotate the model as follows, we will create a pipeline stage per
iteration block:

```python
# Add a split point after each iter_block
annotate_split_points(
    model,
    {"iter_block": PipeSplitWrapper.SplitPoint.END},
)
```

That is, PiPPy would create a split point every time it sees "self.iter_block".

Run it with 4 ranks:
```
$ torchrun --nproc-per-node 4 pippy_unroll.py
```

Print-out of the pipe:
```
************************************* pipe *************************************
GraphModule(
  (submod_0): PipeStageModule(
    (L__self___iter_block_mod_lin): Linear(in_features=512, out_features=512, bias=True)
  )
  (submod_1): PipeStageModule(
    (L__self___iter_block_mod_lin): Linear(in_features=512, out_features=512, bias=True)
  )
  (submod_2): PipeStageModule(
    (L__self___iter_block_mod_lin): Linear(in_features=512, out_features=512, bias=True)
  )
  (submod_3): PipeStageModule(
    (L__self___output_proj): Linear(in_features=512, out_features=10, bias=True)
  )
)

def forward(self, arg0):
    submod_0 = self.submod_0(arg0);  arg0 = None
    submod_1 = self.submod_1(submod_0);  submod_0 = None
    submod_2 = self.submod_2(submod_1);  submod_1 = None
    submod_3 = self.submod_3(submod_2);  submod_2 = None
    return [submod_3]
```
We can see 4 stages as expected (3 iterations plus 1 output projection).

If we print one of the stages, we can see that it contains the code of one iteration:
```
*********************************** submod0 ************************************
PipeStageModule(
  (L__self___iter_block_mod_lin): Linear(in_features=512, out_features=512, bias=True)
)

def forward(self, l_x_):
    l__self___iter_block_mod_lin = self.L__self___iter_block_mod_lin(l_x_);  l_x_ = None
    relu = torch.relu(l__self___iter_block_mod_lin);  l__self___iter_block_mod_lin = None
    return relu
```

## How can this functionality help?
Increase throughput of your model.

Imagine your for loop needs to iterate on the data for `n` times, and it takes time `t` to process 1 sample (yielding a throughput of `1/t`). If we were to unroll the for loop onto `n` devices, then we can push `n` microbatches into the pipeline, each microbatch containing 1 sample. Then at any timeslot, the pipeline is processing `n` samples, yielding a throughput of `n/t`.
