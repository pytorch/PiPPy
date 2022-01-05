import torch
import torch.fx
from torch.fx.passes.split_module import split_module

import copy, operator
from typing import Dict, List, Optional, Union, cast
from enum import Enum

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

_pipeline_tracer = None

def pipe_split():
    if _pipeline_tracer is not None:
        _pipeline_tracer.graph.call_function(pipe_split, (), {})

class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2

MultiUseParamSpec = Union[MultiUseParameterConfig, Dict[str, MultiUseParameterConfig]]

class Pipe(torch.nn.Module):
    def __init__(self, split_gm : torch.fx.GraphModule):
        super().__init__()
        self.split_gm : torch.fx.GraphModule = split_gm

        for node in split_gm.graph.nodes:
            assert (node.op in {'call_module', 'placeholder', 'output'} or 
                (node.op, node.target) == ('call_function', operator.getitem))

        # Detect replicated parameters so we know that we have to do an additional allreduce
        # before applying the optimizer
        #
        # Note that this also handles the case where there were multiple calls to a single
        # module from different stages, regardless of whether that module invocation
        # was handled by the logic above.

        # Map parameter value to a dictionary that maps the user pipeline module
        # to the local qualname within that module
        params_to_users : Dict[torch.nn.Parameter, Dict[str, str]] = {}

        for m_qualname, mod in self.split_gm.named_children():
            for p_qualname, param in mod.named_parameters():
                params_to_users.setdefault(param, {})
                params_to_users[param][m_qualname] = p_qualname

        self.replicated_params : List[Dict[str, str]] = [
            use_mapping for _, use_mapping in params_to_users.items() if len(use_mapping) > 1]

    def forward(self, *args, **kwargs):
        return self.split_gm(*args, **kwargs)

    @staticmethod
    def from_sequential(seq : torch.nn.Sequential):
        # Deduplicate contained modules so we have a unique instance for each
        # call in topological ordering. This is necessary so that detection of shared
        # parameters across stages works.
        new_seq_modules = []

        seen_modules = {}
        for module in seq:
            if module in seen_modules:
                to_append = copy.copy(module)
            else:
                to_append = module
            new_seq_modules.append(to_append)
            seen_modules.setdefault(to_append)

        to_trace = torch.nn.Sequential(*new_seq_modules)

        assert isinstance(seq, torch.nn.Sequential)
        class AllModTracer(torch.fx.Tracer):
            def is_leaf_module(self, *args, **kwargs):
                return True

        tracer = AllModTracer()
        graph = tracer.trace(to_trace)
        gm = torch.fx.GraphModule(tracer.root, graph)

        return Pipe(gm)

    @staticmethod
    def from_tracing(mod : torch.nn.Sequential, multi_use_param_spec : Optional[MultiUseParamSpec] = None):
        # TODO: abstract partitioning policy

        global _pipeline_tracer
        old__pipeline_tracer = _pipeline_tracer
        _pipeline_tracer = torch.fx.Tracer()
        try:
            # TODO: tracing policy
            graph = _pipeline_tracer.trace(mod)
            traced = torch.fx.GraphModule(mod, graph)
        finally:
            _pipeline_tracer = old__pipeline_tracer

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
        def delete_user_reference(node, user, delete_node=True):
            assert len(user.kwargs) == 0
            use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
            assert len(use_idxs) == 1
            args_copy = list(user.args)
            args_copy.pop(use_idxs[0])
            user.args = tuple(args_copy)
            if delete_node:
                node.graph.erase_node(node)

            return use_idxs[0]

        def move_param_to_callee(callee, param_val, use_idx):
            new_param_name = f"__{node.target.replace('.', '_')}"
            assert not hasattr(callee, new_param_name)
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

            return get_attr

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

        # # Handle multi-use parameters based on user's configuration
        multi_use_param_spec = multi_use_param_spec or {}

        multi_use_params_qualnames : Dict[str, Optional[MultiUseParameterConfig]] = {}
        for node in split.graph.nodes:
            if node.op == 'get_attr' and len(node.users) > 1:
                multi_use_params_qualnames.setdefault(node.target)

        for param in multi_use_params_qualnames:
            if isinstance(multi_use_param_spec, MultiUseParameterConfig):
                multi_use_params_qualnames[param] = multi_use_param_spec
            elif isinstance(multi_use_param_spec, dict):
                multi_use_params_qualnames[param] = multi_use_param_spec.get(param, MultiUseParameterConfig.TRANSMIT)
            else:
                raise ValueError('multi_use_param_spec must be MultiUseParamSpec enum or dict')

        multi_use_params_qualnames = cast(Dict[str, Optional[MultiUseParameterConfig]], multi_use_params_qualnames)

        # TODO: do we maintain the invariant that `Node.users` is topologically ordered? I don't think so
        node_to_first_user : Dict[torch.fx.Node, torch.fx.Node] = {}
        for node in split.graph.nodes:
            for input in node.all_input_nodes:
                if input not in node_to_first_user:
                    node_to_first_user[input] = node

        for node in split.graph.nodes:
            if node.op == 'get_attr' and node.target in multi_use_params_qualnames:
                reuse_type = multi_use_params_qualnames[node.target]
                if reuse_type == MultiUseParameterConfig.TRANSMIT:
                    first_user = node_to_first_user[node]
                    assert first_user.op == 'call_module'

                    use_idx = delete_user_reference(node, first_user, delete_node=False)

                    param_val = split
                    for atom in node.target.split('.'):
                        param_val = getattr(param_val, atom)

                    submod = split.get_submodule(first_user.target)

                    callee_param_def = move_param_to_callee(submod, param_val, use_idx)

                    # Add extra output to the callee and switch references to the parameter
                    # access in the pipeline graph to use this.
                    callee_output_nodes = [n for n in submod.graph.nodes if n.op == 'output']
                    assert len(callee_output_nodes) == 1
                    callee_output_node = callee_output_nodes[0]

                    # TODO: zero outputs?
                    if isinstance(callee_output_node.args[0], tuple):
                        new_output_args = callee_output_node.args[0] + (callee_param_def,)
                        callee_output_node.args = (new_output_args,)
                        new_output_idx = len(new_output_args) - 1                        
                        promoted_to_tuple = False
                    else:
                        new_output_args = (callee_output_node.args[0], callee_param_def)
                        callee_output_node.args = (new_output_args,)
                        new_output_idx = len(new_output_args) - 1                        
                        promoted_to_tuple = True

                    submod.graph.lint()
                    submod.recompile()

                    with split.graph.inserting_after(first_user):
                        if promoted_to_tuple:
                            # TODO: test this code path
                            orig_output_getitem = split.graph.call_function(operator.getitem, (first_user, 0))
                            first_user.replace_all_uses_with(orig_output_getitem)

                        transmitted_value_getitem = split.graph.call_function(
                            operator.getitem, (first_user, new_output_idx))
                        node.replace_all_uses_with(transmitted_value_getitem)
                        split.graph.erase_node(node)
                elif reuse_type == MultiUseParameterConfig.REPLICATE:
                    for user in copy.copy(node.users):
                        use_idx = delete_user_reference(node, user, delete_node=False)
                        param_val = split
                        for atom in node.target.split('.'):
                            param_val = getattr(param_val, atom)
                        submod = split.get_submodule(user.target)
                        param_access_node = move_param_to_callee(submod, param_val, use_idx)

                    split.graph.erase_node(node)
                else:
                    raise ValueError(f'Unknown multi-use config value {reuse_type} specified for {node.target}')

        split.graph.lint()
        split.recompile()

        return Pipe(split)


# Test sequential
mods = [torch.nn.Linear(512, 512) for _ in range(5)]
mods += [mods[0]]
seq = torch.nn.Sequential(*mods)

seq_pipe = Pipe.from_sequential(seq)
assert seq_pipe.replicated_params == [{'0': 'weight', '5': 'weight'}, {'0': 'bias', '5': 'bias'}]

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

ec_pipe = Pipe.from_tracing(ec, MultiUseParameterConfig.TRANSMIT)
x = torch.randn(5, 512)
torch.testing.assert_allclose(ec(x), ec_pipe(x))
assert ec_pipe.replicated_params == [
    {'submod_1': 'lin.weight', 'submod_2': 'lin.weight'}, {'submod_1': 'lin.bias', 'submod_2': 'lin.bias'}]

ec_pipe_replicated = Pipe.from_tracing(ec, MultiUseParameterConfig.REPLICATE)
x = torch.randn(5, 512)
torch.testing.assert_allclose(ec(x), ec_pipe_replicated(x))
assert ec_pipe_replicated.replicated_params == [
    {'submod_0': '__mm_param', 'submod_1': '__mm_param'},
    {'submod_1': 'lin.weight', 'submod_2': 'lin.weight'},
    {'submod_1': 'lin.bias', 'submod_2': 'lin.bias'}]


# TODO:
# 1. Test autograd on single-box
# 2. investigate gradient sync for shared parameters. how does DDP do it?
# 3. Modify serialization to put parameters back in their original qualname form?
# . Shape specialized tracing?
# . Can we define semantics for shared module call? Can we make this configurable in the same way as
#    with shared parameters? probably need to modify split_module in this case
# . Add parameter movement to split_module
