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

class Pipe(torch.nn.Module):
    def __init__(self, split_gm : torch.fx.GraphModule):
        super().__init__()
        self.split_gm = split_gm

    def forward(self, *args, **kwargs):
        return self.split_gm(*args, **kwargs)

    @staticmethod
    def from_sequential(seq : torch.nn.Sequential):
        assert isinstance(seq, torch.nn.Sequential)
        class AllModTracer(torch.fx.Tracer):
            def is_leaf_module(self, *args, **kwargs):
                return True

        tracer = AllModTracer()
        graph = tracer.trace(seq)
        gm = torch.fx.GraphModule(tracer.root, graph)
        return Pipe(gm)

    @staticmethod
    def from_tracing(mod : torch.nn.Sequential):
        # TODO: abstract partitioning policy

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

        # TODO: what does split do with module invocations? does it move the modules
        # into the submodules?
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
        def delete_user_reference(node, user):
            assert len(user.kwargs) == 0
            use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
            assert len(use_idxs) == 1
            args_copy = list(user.args)
            args_copy.pop(use_idxs[0])
            user.args = args_copy
            node.graph.erase_node(node)

            return use_idxs[0]

        def move_param_to_callee(callee, param_val, use_idx):
            new_param_name = f"__{node.target.replace('.', '_')}"
            setattr(callee, new_param_name, param_val)

            ph_counter = 0
            for sn in callee.graph.nodes:
                if sn.op == 'placeholder':
                    if ph_counter == use_idx:
                        with callee.graph.inserting_before(sn):
                            get_attr = callee.graph.get_attr(new_param_name)
                            sn.replace_all_uses_with(get_attr)
                            callee.graph.erase_node(sn)
                    ph_counter += 1
            callee.graph.lint()
            callee.recompile()

        for node in split.graph.nodes:
            if node.op == 'get_attr' and len(node.users) == 1:
                user = list(node.users)[0]
                if user.op != 'call_module':
                    continue
                use_idx = delete_user_reference(node, user)

                # Move parameter into submodule and replace PH with a get_attr
                param_val = split
                for atom in node.target.split('.'):
                    param_val = getattr(param_val, atom)

                callee = split.get_submodule(user.target)
                move_param_to_callee(callee, param_val, use_idx)

        split.graph.lint()
        split.recompile()

        return Pipe(split)


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
    self.lin = torch.nn.Linear(512, 512)

  def forward(self, x):
    x = torch.mm(x, self.mm_param)
    skip_connection = x
    x = torch.relu(x)
    pipe_split()
    x = torch.mm(x, self.mm_param)
    x = self.lin(x)
    pipe_split()
    x = torch.relu(x)
    x = x + skip_connection
    x = torch.mm(x, self.mm_param2)
    x = self.lin(x)
    return x

ec = ExampleCode()
ec(torch.randn(50, 512))

ec_pipe = Pipe.from_tracing(ec)

print(ec_pipe.split_gm)
# TODO: split_module replicates reference to submodules but transmits parameters
print(ec_pipe.split_gm.submod_2)

# TODO:
# 1. Shared parameters: configure to either lift into first use and transmit or replicate
# 2. Add parameter movement to split_module
# 3. Generalize split_module to configure the behavior of module calls


x = torch.randn(5, 512)
torch.testing.assert_allclose(ec(x), ec_pipe(x))
