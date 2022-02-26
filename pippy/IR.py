import torch
import torch.fx
from torch.fx.passes.split_module import split_module

import copy
import operator
from typing import Callable, Dict, List, Optional, Union, cast
from enum import Enum
import itertools

# TODO:
# 1. investigate gradient sync for shared parameters. how does DDP do it?
# 2. Modify serialization to put parameters back in their original qualname form?
# 3. Shape specialized tracing?
# 4. Can we define semantics for shared module call? Can we make this configurable in the same way as
#    with shared parameters? probably need to modify split_module in this case
# 5. Add parameter movement to split_module

# TODO: move to a separate file with runtime artifacts
def stage_backward(stage_output, output_grads, input_values):
    """
    Given the input value(s) and the corresponding gradient for those/that input
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """
    torch.autograd.backward(stage_output, grad_tensors=output_grads)

    grad_inputs = []
    for val in input_values:
        if isinstance(val, torch.Tensor):
            grad_inputs.append(val.grad)
        elif val is None:
            grad_inputs.append(None)
        else:
            raise NotImplementedError(f'Cannot pass gradient for value {val}')
    return grad_inputs

def _insert_stage_symbolic_backward(g : torch.fx.Graph):
    output_nodes = [n for n in g.nodes if n.op == 'output']
    assert len(output_nodes) == 1
    output_node = output_nodes[0]

    loss_nodes = [n for n in g.nodes if (n.op, n.target) == ('call_module', '_loss')]
    assert len(loss_nodes) == 1
    loss_node = loss_nodes[0]

    val_to_grad = {loss_node : None}

    def assign_or_accumulate_grad(forward_node, grad_value):
        if forward_node in val_to_grad:
            grad_value = g.call_function(torch.add, (val_to_grad[forward_node], grad_value))
        val_to_grad[forward_node] = grad_value

    with g.inserting_before(output_node):
        for node in reversed(g.nodes):
            if node.op == 'call_module':
                grad_call = g.call_function(stage_backward, kwargs={
                    'stage_output' : node,
                    'output_grads' : val_to_grad[node],
                    'input_values' : list(node.all_input_nodes)
                })

                input_nodes = list(node.all_input_nodes)
                grad_call_proxy = torch.fx.Proxy(grad_call)
                for i, input_node in enumerate(input_nodes):
                    assign_or_accumulate_grad(input_node, grad_call_proxy[i].node)
            elif node.op == 'call_function':
                assert node.target == operator.getitem
                assert len(node.args) == 2
                indexed_value, node_idx = tuple(node.args)

                grad_output = val_to_grad[node]

                # indexed_value is a collection that we are indexing into. It could
                # exist in the val_to_grad map if we've processed another `getitem`
                # already.
                existing_list_size = len(val_to_grad[indexed_value]) if indexed_value in val_to_grad else -1
                new_list_size = max(node_idx + 1, existing_list_size)

                reconstructed_list = [None for _ in range(new_list_size)]

                # Copy over existing elements if present
                if indexed_value in val_to_grad:
                    for i, val in enumerate(val_to_grad[indexed_value]):
                        reconstructed_list[i] = val

                # Populate value represented by this node
                reconstructed_list[node_idx] = grad_output

                val_to_grad[indexed_value] = reconstructed_list

    return g


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
    # hasattr(_pipeline_tracer, 'graph') is a workaround to support HFTracer
    if _pipeline_tracer is not None and hasattr(_pipeline_tracer, 'graph'):
        _pipeline_tracer.graph.call_function(pipe_split, (), {})

class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2

MultiUseParamSpec = Union[MultiUseParameterConfig, Dict[str, MultiUseParameterConfig]]

class DetachExecutor(torch.fx.Interpreter):
    """
    Special interpreter to run the split_gm in testing that detaches all inputs to
    a module invocation. This is needed so that the values at the boundary are
    leaf modules in autograd execution.
    """
    def __init__(self, module, garbage_collect_values=True):
        garbage_collect_values = False
        super().__init__(module, garbage_collect_values)
        self.value_remap = {}

    def run(self, *args, initial_env=None):
        self.value_remap = {}
        return super().run(*args, initial_env=initial_env)

    def call_module(self, target, args, kwargs):
        def detach_tensors(a):
            if isinstance(a, torch.Tensor) and a.requires_grad:
                if a not in self.value_remap:
                    new_val = a.detach().requires_grad_(True)
                    self.value_remap[a] = new_val
                return self.value_remap[a]
            else:
                return a

        args = torch.fx.node.map_aggregate(args, detach_tensors)
        kwargs = torch.fx.node.map_aggregate(kwargs, detach_tensors)

        return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        # HACK to reroute saved input tensors to point to the detach()ed version
        if target == stage_backward:
            kwargs = dict(kwargs)
            kwargs['input_values'] = [self.value_remap.get(v, v) for v in kwargs['input_values']]
        return super().call_function(target, args, kwargs)


class Pipe(torch.nn.Module):
    def __init__(self, split_gm : torch.fx.GraphModule, qualname_mapping : Dict[str, str],
                 num_stages : int, has_loss_and_backward : bool):
        super().__init__()
        self.split_gm : torch.fx.GraphModule = split_gm
        self.executor : DetachExecutor = DetachExecutor(self.split_gm)
        self.num_stages : int = num_stages
        self.has_loss_and_backwards = has_loss_and_backward

        for node in split_gm.graph.nodes:
            assert (node.op in {'call_module', 'placeholder', 'output'} or 
                   (node.op, node.target) == ('call_function', operator.getitem) or
                   (node.op, node.target) == ('call_method', 'backward') or
                   (node.op, node.target) == ('call_function', stage_backward) or
                   (node.op, node.target) == ('call_function', torch.add))

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

        # We must break the aliasing relationship between the replicated parameters for correct
        # numerics in reference runs. If we do not do this, the autograd tape in separate stages
        # will have a reference to the same tensor value and will erroneously apply gradient
        # updates multiple times. Therefore, for each replicated parameter set, we deepcopy the
        # values so that we have separate instances.
        for param_mapping in self.replicated_params:
            for submod_name, param_qualname in param_mapping.items():
                submod = getattr(self.split_gm, submod_name)
                atoms = param_qualname.split('.')
                for atom in atoms[:-1]:
                    submod = getattr(submod, atom)
                setattr(submod, atoms[-1], copy.deepcopy(getattr(submod, atoms[-1])))

        self.new_to_old_qualname_mapping = qualname_mapping

        def throw(self, *args, **kwargs):
            raise RuntimeError('To run pipeline locally, invoke the Pipe object directly, not `split_gm`')
        self.split_gm.forward = throw

    def forward(self, *args, **kwargs):
        executor_args = args
        if len(kwargs) > 0:
            from inspect import Signature, Parameter
            parameters = []
            for node in self.split_gm.graph.nodes:
                if node.op == 'placeholder':
                    if node.args and len(node.args) > 0:
                        parameters.append(Parameter(node.target, Parameter.POSITIONAL_OR_KEYWORD, default=node.args[0]))
                    else:
                        parameters.append(Parameter(node.target, Parameter.POSITIONAL_OR_KEYWORD))
            signature = Signature(parameters)
            ba = signature.bind(*args, **kwargs)
            ba.apply_defaults()
            executor_args = ba.arguments.values()
        return self.executor.run(*executor_args)

    def remap_qualname(self, qualname):
        # TODO: annoying
        if qualname.startswith('split_gm.'):
            qualname = qualname[len('split_gm.'):]
        return self.new_to_old_qualname_mapping[qualname]

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
    def _number_and_count_forward_stages(gm : torch.fx.GraphModule):
        num_stages = 0
        found_idxs = {}
        for node in gm.graph.nodes:
            if node.op == 'call_module' and node.target.startswith('submod_'):
                node.meta['stage_idx'] = int(node.target[len('submod_'):])
                found_idxs.setdefault(node.meta['stage_idx'])
                num_stages += 1

        assert all(i in found_idxs for i in range(num_stages))

        return num_stages

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
        gm.graph.erase_node(output_node)
        gm.graph.output(loss_node)

        # TODO: make this configurable. Do users want to call forward + loss w/o backward?
        _insert_stage_symbolic_backward(gm.graph)

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

                # All submodules should be unique, even if they were aliased in the
                # original Sequential, since we did a shallow copy of all modules above
                assert new_target not in copied_root_dict
                copied_root_dict[new_target] = copied_root_dict.pop(node.target)

                node.target = new_target

        gm = torch.fx.GraphModule(copied_root_dict, graph)

        num_stages = Pipe._number_and_count_forward_stages(gm)

        if loss_fn is not None:
            Pipe._append_traced_loss_fn_to_gm(gm, loss_fn)
            has_loss_and_backward = True
        else:
            has_loss_and_backward = False

        return Pipe(gm, Pipe._hack_build_qualname_mapping(old=seq, new=gm), num_stages, has_loss_and_backward)

    @staticmethod
    def _from_traced(mod: torch.nn.Module, traced: torch.fx.GraphModule,
                     multi_use_param_spec: Optional[MultiUseParamSpec] = None,
                     loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, **kwargs):
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
            new_param_name = f"moved_{node.target.replace('.', '_')}"
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
                assert user.op == 'call_module'
                use_idx = delete_user_reference(node, user)

                # Move parameter into submodule and replace PH with a get_attr
                param_val = split
                for atom in node.target.split('.'):
                    param_val = getattr(param_val, atom)

                callee = split.get_submodule(user.target)
                move_param_to_callee(callee, param_val, use_idx)

                atoms = node.target.split('.')
                mod_itr = split
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)

                delattr(mod_itr, atoms[-1])

        split.graph.lint()
        split.recompile()

        # Handle multi-use parameters based on user's configuration
        # TODO: generalize this to sequential
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

                    atoms = node.target.split('.')
                    mod_itr = split
                    for atom in atoms[:-1]:
                        mod_itr = getattr(mod_itr, atom)

                    delattr(mod_itr, atoms[-1])

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
                            # HACK because the above replace_all_uses with ALSO replaced the instance
                            # of first_user within the getitem node we just added
                            orig_output_getitem.args = (first_user,) + orig_output_getitem.args[1:]

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

                    atoms = node.target.split('.')
                    mod_itr = split
                    for atom in atoms[:-1]:
                        mod_itr = getattr(mod_itr, atom)

                    delattr(mod_itr, atoms[-1])

                    split.graph.erase_node(node)
                else:
                    raise ValueError(f'Unknown multi-use config value {reuse_type} specified for {node.target}')

        split.delete_all_unused_submodules()

        split.graph.lint()
        split.recompile()

        num_stages = Pipe._number_and_count_forward_stages(split)

        if loss_fn is not None:
            Pipe._append_traced_loss_fn_to_gm(split, loss_fn)
            has_loss_and_backward = True
        else:
            has_loss_and_backward = False

        return Pipe(split, Pipe._hack_build_qualname_mapping(old=mod, new=split), num_stages, has_loss_and_backward)

    @staticmethod
    def from_tracing(mod : torch.nn.Module, multi_use_param_spec : Optional[MultiUseParamSpec] = None,
                     loss_fn : Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                     tracer=None, **kwargs):
        # TODO: abstract partitioning policy

        # TODO(pbelevich): Here we can add support for custom torch.fx tracers
        #  https://github.com/jamesr66a/PiPPy/issues/15

        global _pipeline_tracer
        old__pipeline_tracer = _pipeline_tracer
        _pipeline_tracer = tracer or torch.fx.Tracer()
        try:
            # TODO: tracing policy
            graph = _pipeline_tracer.trace(mod, **kwargs)
            traced = torch.fx.GraphModule(mod, graph)
        finally:
            _pipeline_tracer = old__pipeline_tracer

        return Pipe._from_traced(mod, traced, multi_use_param_spec, loss_fn, **kwargs)


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
        for i, atom in enumerate(atoms[:-1]):
            try:
                predecessor_module = getattr(predecessor_module, atom)
            except AttributeError as e:
                raise AttributeError(f'Specified target {qualname} referenced nonexistent module {".".join(atoms[:i+1])}')

        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        wrapped_mod = PipeSplitWrapper(mod_to_wrap, split_type)
        setattr(predecessor_module, atoms[-1], wrapped_mod)
