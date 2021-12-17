import torch
import torch.fx
from torch.fx.passes.split_module import split_module

from typing import Callable, Dict, Optional, Tuple

# Pipe model representation
#
# Pipe can be thought of as an `nn.Sequential++`. That is to say: it specifies
# a single topological ordering of pipeline "stages" that, when run in series,
# constitutes all of the operations of the program. However, unlike `nn.Sequential`,
# Pipe allows non-local usages of values, so long as those uses still respect
# topological ordering. In particular:
#
# 1. Non-local activations. This type of usage can appear in, for example, skip
#    connections. These values will be directly transmitted from the "def" stage
#    to all stages that use them skipping intermediate stages. During autograd,
#    gradients will be propagated back through this skip connection reverse
#    to how activations propagated in the forward pass.
# 2. Non-local parameter/module invocations. This occurs when a parameter is used
#    in a stage downstream of where it is resident. These values can be carried
#    forward similarly to (1), but in addition one might want to replicate the
#    value on multiple stages. Gradients for these shared parameters will be
#    accumulated separately on each stage, but there will be an additional
#    gradient accumulation before the optimizer step.

pipeline_tracer = None

def pipe_split():
    if pipeline_tracer is not None:
        pipeline_tracer.graph.call_function(pipe_split, (), {})

class PipeStage(torch.nn.Module):
    def __init__(self, module : torch.nn.Module):
        super().__init__()
        self.module = module
        self.sends = {}
        self.receives = {}

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class Pipe(torch.nn.Module):
    def __init__(self, stages : Dict[str, PipeStage]):
        super().__init__()
        self.stages = stages

    def forward(self, *args, **kwargs):
        for name, stage in self.stages.items():
            # TODO: generalize this calling convention
            rv = stage(*args, **kwargs)
            args = (rv,)
            kwargs = {}
        return args[0]

    @staticmethod
    def from_sequential(seq : torch.nn.Sequential):
        stages = {str(i): PipeStage(mod) for i, mod in enumerate(seq)}
        return Pipe(stages)

    @staticmethod
    def from_tracing(mod : torch.nn.Sequential):
        # TODO: partitioning policy


        global pipeline_tracer
        old_pipeline_tracer = pipeline_tracer
        pipeline_tracer = torch.fx.Tracer()
        try:
            # TODO: tracing policy
            graph = pipeline_tracer.trace(mod)
            traced = torch.fx.GraphModule(mod, graph)
        finally:
            pipeline_tracer = old_pipeline_tracer

        part_idx = 0
        def split_callback(n : torch.fx.Node):
            nonlocal part_idx
            if (n.op, n.target) == ('call_function', pipe_split):
                part_idx += 1
            return part_idx

        split = split_module(traced, mod, split_callback)

        # peephole to remove pipe_split
        for submodule in split.modules():
            if isinstance(submodule, torch.fx.GraphModule):
                for node in submodule.graph.nodes:
                    if (node.op, node.target) == ('call_function', pipe_split):
                        submodule.graph.erase_node(node)
                submodule.recompile()

        # lift single-use parameter fetches into the modules that use them
        # TODO: backport this into split_module

        print(split)
        import pdb; pdb.set_trace()

        # TODO: lol
        return Pipe({'0': PipeStage(traced)})


# Test sequential
mods = [torch.nn.Linear(512, 512) for _ in range(5)]
seq = torch.nn.Sequential(*mods)

seq_pipe = Pipe.from_sequential(seq)

x = torch.randn(50, 512)
torch.testing.assert_allclose(seq(x), seq_pipe(x))


# Test partitioning and skip connection

class ExampleCode(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mm_param = torch.nn.Parameter(torch.randn(512, 512))
    self.mm_param2 = torch.nn.Parameter(torch.randn(512, 512))

  def forward(self, x):
    x = torch.mm(x, self.mm_param)
    pipe_split()
    skip_connection = x
    x = torch.relu(x)
    pipe_split()
    x = torch.mm(x, self.mm_param)
    pipe_split()
    x = torch.relu(x)
    x = x + skip_connection
    x = torch.mm(x, self.mm_param2)
    return x

ec = ExampleCode()
ec(torch.randn(50, 512))

ec_pipe = Pipe.from_tracing(ec)

x = torch.randn(5, 512)
torch.testing.assert_allclose(ec(x), ec_pipe(x))
