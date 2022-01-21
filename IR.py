from termios import ECHOE
import torch
import torch.fx
from torch.fx.passes.split_module import split_module

import copy, operator
from typing import Callable, Dict, List, Optional, Union, cast
from enum import Enum
import itertools

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
    def __init__(self, split_gm : torch.fx.GraphModule, qualname_mapping : Dict[str, str]):
        super().__init__()
        self.split_gm : torch.fx.GraphModule = split_gm

        for node in split_gm.graph.nodes:
            assert (node.op in {'call_module', 'placeholder', 'output'} or 
                (node.op, node.target) == ('call_function', operator.getitem) or
                (node.op, node.target) == ('call_method', 'backward'))

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

        self.new_to_old_qualname_mapping = qualname_mapping

    def forward(self, *args, **kwargs):
        return self.split_gm(*args, **kwargs)

    def remap_qualname(self, qualname):
        # TODO: annoying
        if qualname.startswith('split_gm.'):
            qualname = qualname[len('split_gm.'):]
        return self.new_to_old_qualname_mapping.get(qualname, qualname)

    @staticmethod
    def _hack_build_qualname_mapping(old : torch.nn.Module, new : torch.nn.Module):
        # HACK: this derives a mapping of qualified names from the split module
        # to the orignal module by inspecting the values present in the two
        # modules. Ideally we would track this information while building the
        # new module, which would probably involve modifying split_module
        # to return its internal `Partition.targets` mapping. I am currently
        # too lazy to do this the right way so we are left with this hack.
        new_mod_qn_to_values = {k : v for k, v in itertools.chain(new.named_parameters(), new.named_modules())}
        # Multiple qualnames may map to the same value, so we record potentially multiple qualnames
        # for each value
        old_values_to_qns = {}
        for k, v in itertools.chain(old.named_parameters(), old.named_modules()):
            old_values_to_qns.setdefault(v, {})
            old_values_to_qns[v].setdefault(k)

        new_to_old_mapping = {}
        for k, v in new_mod_qn_to_values.items():
            if v in old_values_to_qns:
                old_qns = old_values_to_qns[v]
                for old_qn in old_qns:
                    new_to_old_mapping[k] = old_qn

        return new_to_old_mapping

    @staticmethod
    def _append_traced_loss_fn_to_gm(gm : torch.fx.GraphModule, loss_fn : Callable[[torch.Tensor], torch.Tensor]):
        last_ph_node = None
        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                last_ph_node = node

        assert last_ph_node is not None
        with gm.graph.inserting_after(last_ph_node):
            target_ph_node = gm.graph.placeholder('target')

        output_node = None
        for node in gm.graph.nodes:
            if node.op == 'output':
                output_node = node
                break

        if isinstance(loss_fn, torch.nn.Module):
            assert not hasattr(gm, '_loss')
            gm.add_module('_loss', loss_fn)
        else:
            # Trace loss computation into a new _loss submodule
            # TODO: configurable tracing
            traced_loss_submod = torch.fx.symbolic_trace(loss_fn)
            assert not hasattr(gm, '_loss')
            gm.add_module('_loss', traced_loss_submod)

        assert output_node is not None
        assert len(output_node.args) == 1
        output_val = output_node.args[0]
        with gm.graph.inserting_after(output_node):
            loss_node = gm.graph.call_module('_loss', (output_val, target_ph_node))
        # TODO: make this configurable. Do users want to call forward + loss w/o backward?
        with gm.graph.inserting_after(loss_node):
            gm.graph.call_method('backward', (loss_node,))
        gm.graph.erase_node(output_node)
        gm.graph.output(loss_node)
        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def from_sequential(seq : torch.nn.Sequential, loss_fn : Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
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
        # Rename top-level submodules to align with the qualnames produced
        # by split_module in the `from_tracing` frontend

        copied_root_dict = dict(tracer.root.named_modules())

        for node in graph.nodes:
            if node.op == 'call_module':
                new_target = f'submod_{node.target}'

                # submod may have already been renamed by a previous call_module
                if node.target in copied_root_dict:
                    copied_root_dict[new_target] = copied_root_dict.pop(node.target)
                else:
                    assert new_target in copied_root_dict

                node.target = new_target

        gm = torch.fx.GraphModule(copied_root_dict, graph)

        if loss_fn is not None:
            Pipe._append_traced_loss_fn_to_gm(gm, loss_fn)

        return Pipe(gm, Pipe._hack_build_qualname_mapping(old=seq, new=gm))

    @staticmethod
    def from_tracing(mod : torch.nn.Module, multi_use_param_spec : Optional[MultiUseParamSpec] = None,
                     loss_fn : Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, **kwargs):
        # TODO: abstract partitioning policy

        global _pipeline_tracer
        old__pipeline_tracer = _pipeline_tracer
        _pipeline_tracer = torch.fx.Tracer()
        try:
            # TODO: tracing policy
            graph = _pipeline_tracer.trace(mod, **kwargs)
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

        if loss_fn is not None:
            Pipe._append_traced_loss_fn_to_gm(split, loss_fn)

        return Pipe(split, Pipe._hack_build_qualname_mapping(old=mod, new=split))


class PipeSplitWrapper(torch.nn.Module):
    class SplitPoint(Enum):
        BEGINNING = 1
        END = 2

    def __init__(self, mod : torch.nn.Module, split_point : SplitPoint = SplitPoint.BEGINNING):
        super().__init__()
        self.mod = mod
        self.split_point = split_point

    def forward(self, *args, **kwargs):
        try:
            if self.split_point == self.SplitPoint.BEGINNING:
                pipe_split()

            return self.mod(*args, **kwargs)
        finally:
            if self.split_point == self.SplitPoint.END:
                pipe_split()


def annotate_split_points(mod : torch.nn.Module, spec : Dict[str, Optional[PipeSplitWrapper.SplitPoint]]):
    for qualname, split_type in spec.items():
        atoms = qualname.split('.')
        predecessor_module = mod
        for atom in atoms[:-1]:
            predecessor_module = getattr(predecessor_module, atom)

        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        wrapped_mod = PipeSplitWrapper(mod_to_wrap, split_type)
        setattr(predecessor_module, atoms[-1], wrapped_mod)


# Test sequential
mods = [torch.nn.Linear(512, 512) for _ in range(5)]
mods += [mods[0]]
seq = torch.nn.Sequential(*mods)

seq_pipe = Pipe.from_sequential(seq)
assert seq_pipe.replicated_params == [{'submod_0': 'weight', 'submod_5': 'weight'}, {'submod_0': 'bias', 'submod_5': 'bias'}]

x = torch.randn(50, 512)
torch.testing.assert_allclose(seq(x), seq_pipe(x))

def check_qualname_mapping(old, new):
    seen_old_qns = {}
    for _, old_qn in new.new_to_old_qualname_mapping.items():
            seen_old_qns.setdefault(old_qn)

    for param_name, _ in old.named_parameters():
        assert param_name in seen_old_qns, f'Expected parameter {param_name} in {seen_old_qns}'

    for mod_name, _ in old.named_modules():
        if mod_name == '':
            continue
        assert mod_name in seen_old_qns,  f'Expected module {mod_name} in {seen_old_qns}'

check_qualname_mapping(old=seq, new=seq_pipe)


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
check_qualname_mapping(old=ec, new=ec_pipe)

ec_pipe_replicated = Pipe.from_tracing(ec, MultiUseParameterConfig.REPLICATE)
x = torch.randn(5, 512)
torch.testing.assert_allclose(ec(x), ec_pipe_replicated(x))
assert ec_pipe_replicated.replicated_params == [
    {'submod_0': '__mm_param', 'submod_1': '__mm_param'},
    {'submod_1': 'lin.weight', 'submod_2': 'lin.weight'},
    {'submod_1': 'lin.bias', 'submod_2': 'lin.bias'}]
check_qualname_mapping(old=ec, new=ec_pipe_replicated)

# TODO:
# 1. Test autograd on single-box
# 2. investigate gradient sync for shared parameters. how does DDP do it?
# 3. Modify serialization to put parameters back in their original qualname form?
# . Shape specialized tracing?
# . Can we define semantics for shared module call? Can we make this configurable in the same way as
#    with shared parameters? probably need to modify split_module in this case
# . Add parameter movement to split_module


# **** Test loss & backward representation - sequential frontend

mse_loss = torch.nn.MSELoss()
seq_pipe_with_loss = Pipe.from_sequential(seq, mse_loss)
check_qualname_mapping(old=seq, new=seq_pipe_with_loss)

test_optim = torch.optim.SGD(seq_pipe_with_loss.parameters(), lr=0.01, momentum=0.9)
ref_optim = torch.optim.SGD(seq.parameters(), lr=0.01, momentum=0.9)

x = torch.randn(5, 512)
target = torch.zeros(5, 512)

test_optim.zero_grad()
test_out = seq_pipe_with_loss(x, target)
test_grads = {seq_pipe_with_loss.remap_qualname(name): copy.copy(val.grad) for name, val in seq_pipe_with_loss.named_parameters()}
torch.testing.assert_allclose(seq_pipe_with_loss(x, target), mse_loss(seq(x), target))

ref_optim.zero_grad()
ref_out = mse_loss(seq(x), target)
ref_out.backward()
ref_grads = {name: copy.copy(val.grad) for name, val in seq.named_parameters()}

for name, ref_grad in ref_grads.items():
    assert name in test_grads
    torch.testing.assert_allclose(test_grads[name], ref_grad)

# **** Test loss & backward representation - tracing frontend

ec_pipe_with_loss = Pipe.from_tracing(ec, loss_fn=mse_loss)
check_qualname_mapping(old=ec, new=ec_pipe_with_loss)

test_optim = torch.optim.SGD(ec_pipe_with_loss.parameters(), lr=0.01, momentum=0.9)
ref_optim = torch.optim.SGD(ec.parameters(), lr=0.01, momentum=0.9)

x = torch.randn(5, 512)
target = torch.zeros(5, 512)

test_optim.zero_grad()
test_out = ec_pipe_with_loss(x, target)
test_grads = {ec_pipe_with_loss.remap_qualname(name): copy.copy(val.grad) for name, val in ec_pipe_with_loss.named_parameters()}
torch.testing.assert_allclose(ec_pipe_with_loss(x, target), mse_loss(ec(x), target))

ref_optim.zero_grad()
ref_out = mse_loss(ec(x), target)
ref_out.backward()
ref_grads = {name: copy.copy(val.grad) for name, val in ec.named_parameters()}
