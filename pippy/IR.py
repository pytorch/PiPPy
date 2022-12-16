# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import logging
import operator
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx as torch_fx
import pippy.fx
from packaging import version
from pippy.fx.passes.split_module import split_module
from pippy.backward import (
    stage_backward,
    sync_barrier,
    _null_coalesce_accumulate,
)

from torchdistx import deferred_init

# Because split_module with 4 arguments is available only in PT 1.12+
TORCH_FX_REQUIRED_VERSION = version.parse("1.12")

torch_version = version.parse(torch.__version__)
assert (torch_version.major, torch_version.minor) >= (  # type: ignore
    TORCH_FX_REQUIRED_VERSION.major,  # type: ignore
    TORCH_FX_REQUIRED_VERSION.minor,  # type: ignore
), "PiPPy requires PyTorch >= 1.12"

# TODO:
# 1. investigate gradient sync for shared parameters. how does DDP do it?
# 2. Modify serialization to put parameters back in their original qualname form?
# 3. Shape specialized tracing?
# 4. Can we define semantics for shared module call? Can we make this configurable in the same way as
#    with shared parameters? probably need to modify split_module in this case
# 5. Add parameter movement to split_module


def _find_loss_from_output_and_spec(output_val, spec_val):
    if spec_val is False:
        return None
    if spec_val is True:
        if not isinstance(output_val, pippy.fx.Node):
            raise RuntimeError(
                f"Loss spec must specify a dynamic value but got {output_val}"
            )
        return output_val

    if isinstance(spec_val, (tuple, list)):
        if not isinstance(output_val, (tuple, list)):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        if len(output_val) != len(spec_val):
            raise RuntimeError(
                f"Output value {output_val} must match length of loss specification "
                f"{spec_val}"
            )
        for out, spec in zip(output_val, spec_val):
            loss_val = _find_loss_from_output_and_spec(out, spec)
            if loss_val is not None:
                return loss_val
        raise RuntimeError(
            f"Did not find loss value in specification {spec_val}"
        )

    if isinstance(spec_val, dict):
        if not isinstance(output_val, dict):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        if set(output_val.keys()) != set(spec_val.keys()):
            raise RuntimeError(
                f"Output value {output_val} must match keys of loss specification "
                f"{spec_val}"
            )
        for k in spec_val:
            loss_val = _find_loss_from_output_and_spec(
                output_val[k], spec_val[k]
            )
            if loss_val is not None:
                return loss_val
        raise RuntimeError(
            f"Did not find loss value in specification {spec_val}"
        )

    raise RuntimeError(
        f"Unsupported type {type(spec_val)} in loss specification"
    )


def _insert_stage_symbolic_backward(g: pippy.fx.Graph, output_loss_value_spec):
    output_nodes = [n for n in g.nodes if n.op == "output"]
    assert len(output_nodes) == 1
    output_node = output_nodes[0]

    if output_loss_value_spec:
        loss_node = _find_loss_from_output_and_spec(
            output_node.args[0], output_loss_value_spec
        )
    else:
        assert len(output_node.args) == 1
        loss_node = output_node.args[0]

    # Collect metadata about tuple output values. TODO: move this to split_module or FX IR
    tuples: Dict[pippy.fx.Node, Tuple] = {}
    for node in reversed(g.nodes):
        if node.op == "call_function":
            # In the forward pass, only emit placeholder, module calls, and
            # getitem calls. If we have a target other than getitem in this
            # (forward-only) code, there is a bug.
            assert node.target == operator.getitem, (
                "Found non-getitem call in forward pass. "
                "Please report a bug to PiPPy"
            )
            assert (
                len(node.args) == 2
            ), "Found malformed getitem call. Please report a bug to PiPPy"
            indexed_value, node_idx = tuple(node.args)

            # indexed_value is a collection that we are indexing into. It could
            # exist in the tuples map if we've processed another `getitem`
            # already.
            existing_list_size = (
                len(tuples[indexed_value]) if indexed_value in tuples else -1
            )
            new_list_size = max(node_idx + 1, existing_list_size)

            reconstructed_list = [None for _ in range(new_list_size)]

            # Copy over existing elements if present
            if indexed_value in tuples:
                for i, val in enumerate(tuples[indexed_value]):
                    reconstructed_list[i] = val

            # Populate value represented by this node
            reconstructed_list[node_idx] = node

            tuples[indexed_value] = tuple(reconstructed_list)

    # Keep track of nodes that dominate the loss node.
    # We will only emit backward operations for nodes that can contribute
    # to the specified loss value.
    live_nodes = {loss_node: None}
    val_to_grad: Dict[pippy.fx.Node, Optional[pippy.fx.Node]] = {
        loss_node: None
    }

    def assign_or_accumulate_grad(forward_node, grad_value):
        if forward_node in val_to_grad and forward_node.op != "placeholder":
            grad_value = g.call_function(
                _null_coalesce_accumulate,
                (val_to_grad[forward_node], grad_value),
            )
        val_to_grad[forward_node] = grad_value

    with g.inserting_before(output_node):
        barrier_tokens = []
        last_grads = None

        for node in reversed(g.nodes):
            if node not in live_nodes:
                continue

            def add_to_live_nodes(n):
                live_nodes.setdefault(n, None)

            pippy.fx.node.map_arg(node.args, add_to_live_nodes)
            pippy.fx.node.map_arg(node.kwargs, add_to_live_nodes)
            if node.op == "call_module":
                output_grads: Union[
                    Tuple[Optional[pippy.fx.Node], ...], Optional[pippy.fx.Node]
                ]
                if node in tuples:
                    stage_output = tuples[node]
                    output_grads = tuple(
                        val_to_grad.get(n, None) for n in tuples[node]
                    )
                    outputs_with_grads_idxs = [
                        i for i, n in enumerate(tuples[node]) if n in live_nodes
                    ]
                else:
                    stage_output = (node,)
                    output_grads = val_to_grad[node]
                    outputs_with_grads_idxs = [0]

                output_grads = (
                    (output_grads,)
                    if not isinstance(output_grads, tuple)
                    else output_grads
                )

                grad_call = g.call_function(
                    stage_backward,
                    kwargs={
                        "stage_output": stage_output,
                        "output_grads": output_grads,
                        "input_values": list(node.all_input_nodes),
                        "outputs_with_grads_idxs": outputs_with_grads_idxs,
                    },
                )
                # Insert backward stage debug info
                kwargs_copy = dict(grad_call.kwargs)
                kwargs_copy[
                    "stage_info"
                ] = f"{grad_call} for stage {node.format_node()}"
                grad_call.kwargs = kwargs_copy

                grad_call_proxy = pippy.fx.Proxy(grad_call)
                grads, barrier_token = (
                    grad_call_proxy[0].node,
                    grad_call_proxy[1].node,
                )
                barrier_tokens.append(barrier_token)
                last_grads = grads

                input_nodes = list(node.all_input_nodes)
                grads_proxy = pippy.fx.Proxy(grads)
                for i, input_node in enumerate(input_nodes):
                    assign_or_accumulate_grad(input_node, grads_proxy[i].node)

        # Insert barrier call - reconnect the original pipeline output (output_node.args[0])
        # to go through the `sync_barrier` call, then make the pipeline output the output
        # of the sync_barrier call. When the driver gets the pipeline output, it is
        # guaranteed that all backwards jobs for that micro-batch have been executed.
        # When all micro-batch pipeline outputs are ready, gradients have been fully
        # computed and synchronized and the optimizer step can be applied.
        barrier_call = g.call_function(
            sync_barrier, (output_node.args[0], barrier_tokens, last_grads)
        )
        output_node.args = (barrier_call,)

    return g


class PipeSequential(torch.nn.Sequential):
    @staticmethod
    def from_sequential(sequential_instance: torch.nn.Sequential):
        return PipeSequential(*[copy.copy(m) for m in sequential_instance])

    def forward(self, input):
        for i, module in enumerate(self):
            input = module(input)
            if i != len(self) - 1:
                pipe_split()
        return input


class LossWrapper(torch.nn.Module):
    """
    LossWrapper is a convenient abstract class that allows you to wrap up both
    your model as well as its loss function and specify the connectivity between
    the inputs, model, loss function, and output value. Example::

        class MyModelWrapper(LossWrapper):
            def forward(self, x, targets):
                model_out = self.module(x)
                loss_value = self.loss_fn(model_out, targets)
                return loss_value

    The above example defines a connectivity where we expect the forward/loss/backward
    training procedure to take two arguments (x and targets), pass x into the module
    to get the output of the feedforward computation, pass the model output and the
    targets value into the loss function, and get and return the loss value, which will
    be backpropagated by PiPPy. The above class would then be instantiated like::

        model = ... # instantiate the model
        loss_fn = torch.nn.MSELoss() # for the sake of demonstration

        wrapper = MyModelWrapper(model, loss_fn)
        pipe = Pipe.from_tracing(wrapper, ...)

    """

    def __init__(self, module, loss_fn):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This instance of LossWrapper does not have an overridden"
            "forward(). Please implement forward() to specify the arguments, "
            "connection between the module and loss, and loss output "
            "value."
        )


class TrivialLossWrapper(LossWrapper):
    def forward(self, x, targets):
        model_out = self.module(x)
        return self.loss_fn(model_out, targets)


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
    if _pipeline_tracer is not None and hasattr(_pipeline_tracer, "graph"):
        _pipeline_tracer.graph.call_function(pipe_split, (), {})


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2


MultiUseParamSpec = Union[
    MultiUseParameterConfig, Dict[str, MultiUseParameterConfig]
]


class DetachExecutor(pippy.fx.Interpreter):
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

        def dont_traverse_size(a):
            return type(a) != torch.Size

        args = pippy.fx.node.map_aggregate(
            args, detach_tensors, dont_traverse_size
        )
        kwargs = pippy.fx.node.map_aggregate(
            kwargs, detach_tensors, dont_traverse_size
        )

        return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        # HACK to reroute saved input tensors to point to the detach()ed version
        if target == stage_backward:
            kwargs = dict(kwargs)
            kwargs["input_values"] = [
                self.value_remap.get(v, v) for v in kwargs["input_values"]
            ]
        return super().call_function(target, args, kwargs)


class _NodeReference:
    def __init__(self, name):
        self.name = name

    name: str


class _LinearNodeList:
    def __init__(self, node_list):
        self.serialize_node_list = []
        for node in node_list:
            node_args = pippy.fx.node.map_arg(
                node.args, lambda n: _NodeReference(n.name)
            )
            node_kwargs = pippy.fx.node.map_arg(
                node.kwargs, lambda n: _NodeReference(n.name)
            )
            serialize_node = pippy.fx.Node(
                graph=None,
                name=node.name,
                op=node.op,
                target=node.target,
                args=node_args,
                kwargs=node_kwargs,
                return_type=node.type,
            )
            serialize_node.meta = copy.copy(node.meta)
            self.serialize_node_list.append(serialize_node)

    def to_graph(self):
        graph = pippy.fx.Graph()

        ref_str_to_node: Dict[str, pippy.fx.Node] = {}

        def ref_to_node(arg):
            if isinstance(arg, _NodeReference):
                return ref_str_to_node[arg.name]
            else:
                return arg

        for node in self.serialize_node_list:
            node_args = pippy.fx.node.map_aggregate(node.args, ref_to_node)
            node_kwargs = pippy.fx.node.map_aggregate(node.kwargs, ref_to_node)
            deser_node = graph.create_node(
                op=node.op,
                target=node.target,
                args=node_args,
                kwargs=node_kwargs,
                name=node.name,
                type_expr=node.type,
            )
            ref_str_to_node[node.name] = deser_node

        return graph


def _direct_serialization_deserialize(body, nodes):
    """
    Custom `__reduce__` method for serialization.
    DO AS I SAY -- NOT AS I DO. This violates the principle that
    GraphModules serialize via code export & re-tracing. We allow
    for this here because **PIPE STAGES SHOULD NOT BE PERSISTED
    TO DISK -- THIS IS ONLY FOR TRANSMISSION VIA RPC**. Persisting
    these instances to disk will expose internal implementation
    details of `fx.Graph` and related data structures and is
    NOT advised.
    """

    class DummyModule(torch.nn.Module):
        def __init__(self, body):
            super().__init__()
            self.__dict__.update(body)

    dummy = DummyModule(body)

    return pippy.fx.GraphModule(dummy, nodes.to_graph())


def _direct_serialization_reduce(self):
    serialization_dict = dict(self.__dict__)
    serialization_dict.pop("_graph")
    return (
        _direct_serialization_deserialize,
        (serialization_dict, _LinearNodeList(self.graph.nodes)),
    )


def _find_mod_device(mod):
    # We assume that all parameters in the module are on the same device
    # HACK: we assume the module has at least one parameter
    param = next(mod.parameters(), None)
    buffer = next(mod.buffers(), None)
    if param is not None:
        device = param.device
    elif buffer is not None:
        device = buffer.device
    else:
        logging.info(
            f"cannot figure out device. Setting it to cpu"
        )
        device = torch.device("cpu")
    return device


class Pipe(torch.nn.Module):
    def __init__(
        self,
        split_gm: pippy.fx.GraphModule,
        qualname_mapping: Dict[str, str],
        num_stages: int,
        has_loss_and_backward: bool,
    ):
        super().__init__()
        self.split_gm: pippy.fx.GraphModule = split_gm
        self.executor: DetachExecutor = DetachExecutor(self.split_gm)
        self.num_stages: int = num_stages
        self.has_loss_and_backwards = has_loss_and_backward
        self.last_grads = None
        self.submod = None

        for node in split_gm.graph.nodes:
            assert (
                node.op in {"call_module", "placeholder", "output"}
                or (node.op, node.target) == ("call_function", operator.getitem)
                or (node.op, node.target) == ("call_method", "backward")
                or (node.op, node.target) == ("call_function", stage_backward)
                or (node.op, node.target)
                == ("call_function", _null_coalesce_accumulate)
                or (node.op, node.target) == ("call_function", sync_barrier)
            ), node

        # Detect replicated parameters so we know that we have to do an additional allreduce
        # before applying the optimizer
        #
        # Note that this also handles the case where there were multiple calls to a single
        # module from different stages, regardless of whether that module invocation
        # was handled by the logic above.

        # Map parameter value to a dictionary that maps the user pipeline module
        # to the local qualname within that module
        params_to_users: Dict[torch.nn.Parameter, Dict[str, str]] = {}

        for m_qualname, mod in self.split_gm.named_children():
            for p_qualname, param in mod.named_parameters():
                params_to_users.setdefault(param, {})
                params_to_users[param][m_qualname] = p_qualname

        self.replicated_params: List[Dict[str, str]] = [
            use_mapping
            for _, use_mapping in params_to_users.items()
            if len(use_mapping) > 1
        ]

        # We must break the aliasing relationship between the replicated parameters for correct
        # numerics in reference runs. If we do not do this, the autograd tape in separate stages
        # will have a reference to the same tensor value and will erroneously apply gradient
        # updates multiple times. Therefore, for each replicated parameter set, we deepcopy the
        # values so that we have separate instances.
        for param_mapping in self.replicated_params:
            for submod_name, param_qualname in param_mapping.items():
                submod = getattr(self.split_gm, submod_name)
                atoms = param_qualname.split(".")
                for atom in atoms[:-1]:
                    submod = getattr(submod, atom)
                setattr(
                    submod, atoms[-1], copy.deepcopy(getattr(submod, atoms[-1]))
                )

        self.new_to_old_qualname_mapping = qualname_mapping

        def throw(self, *args, **kwargs):
            raise RuntimeError(
                "To run pipeline locally, invoke the Pipe object directly, not `split_gm`"
            )

        self.split_gm.forward = throw  # type: ignore

        # Make submodules use custom direct-serialized GraphModule
        i = 0
        while True:
            try:
                name = f"submod_{i}"
                submod = getattr(self.split_gm, name)
                submod.__class__.__reduce__ = _direct_serialization_reduce
                i += 1
            except AttributeError:
                break

    def forward(self, *args, **kwargs):
        executor_args = args
        if len(kwargs) > 0:
            from inspect import Signature, Parameter

            parameters = []
            for node in self.split_gm.graph.nodes:
                if node.op == "placeholder":
                    if node.args and len(node.args) > 0:
                        parameters.append(
                            Parameter(
                                node.target,
                                Parameter.POSITIONAL_OR_KEYWORD,
                                default=node.args[0],
                            )
                        )
                    else:
                        parameter_kind = Parameter.POSITIONAL_OR_KEYWORD
                        param_name = node.target
                        if node.target.startswith("**"):
                            parameter_kind = Parameter.VAR_KEYWORD  # type: ignore[assignment]
                            param_name = param_name[2:]
                        elif node.target.startswith("*"):
                            parameter_kind = Parameter.VAR_POSITIONAL  # type: ignore[assignment]
                            param_name = param_name[1:]
                        parameters.append(Parameter(param_name, parameter_kind))
            signature = Signature(parameters)
            ba = signature.bind(*args, **kwargs)
            ba.apply_defaults()
            executor_args = ba.arguments.values()  # type: ignore[assignment]

        res = self.executor.run(*executor_args)

        if self.has_loss_and_backwards:
            res, self.last_grads = res

        return res

    def remap_qualname(self, qualname):
        # TODO: annoying
        if qualname.startswith("split_gm."):
            qualname = qualname[len("split_gm.") :]

        # The qualname map does not store recursive items, thus,
        # when passed a qualname with leaves, we need to perform longest prefix match
        if qualname not in self.new_to_old_qualname_mapping:
            # Split from the right, one each time
            split_names = qualname.rsplit(".", 1)
            leaf = split_names[-1]
            while len(split_names) > 1:
                prefix = split_names[0]
                if prefix in self.new_to_old_qualname_mapping:
                    old_prefix = self.new_to_old_qualname_mapping[prefix]
                    return ".".join([old_prefix, leaf])
                split_names = prefix.rsplit(".", 1)
                leaf = ".".join([split_names[-1], leaf])

        # Either full name match, or key not found
        return self.new_to_old_qualname_mapping[qualname]

    @staticmethod
    def _number_and_count_forward_stages(gm: pippy.fx.GraphModule):
        num_stages = 0
        found_idxs: Dict[int, None] = {}
        for node in gm.graph.nodes:
            if node.op == "call_module" and node.target.startswith("submod_"):
                node.meta["stage_idx"] = int(node.target[len("submod_") :])
                found_idxs.setdefault(node.meta["stage_idx"])
                num_stages += 1

        # this assert will fail if a split point is inserted before the first layer, which creates empty first submodule
        assert all(i in found_idxs for i in range(num_stages))

        return num_stages

    @staticmethod
    def _from_traced(
        mod: torch.nn.Module,
        traced: pippy.fx.GraphModule,
        multi_use_param_spec: Optional[MultiUseParamSpec] = None,
        output_loss_value_spec=None,
    ):
        """
        Additionally, the ``output_loss_value_spec`` value can be specified to disambiguate
        which value in the output of `forward` is the loss value on which PiPPy should apply
        backpropagation. For example, if your ``forward`` returns a tuple ``(loss, model_out)``,
        you can specify ``output_loss_value_spec=(True, False)``. Or, if your ``forward`` returns
        a dict ``{'loss': loss_value, 'model_out': model_out}``, you can specify
        ``output_loss_value_spec={'loss': True, 'model_out': False}``
        """

        # Deduplicate `get_attr` nodes that refer to the same parameter . Downstream code for moving
        # parameters relies on the invariant that parameter accesses happen once. This is not necessarily
        # the case (especially with custom tracers), so fix that up here.
        get_attr_nodes: Dict[str, pippy.fx.Node] = {}
        for node in traced.graph.nodes:
            if node.op == "get_attr":
                get_attr_nodes.setdefault(node.target, node)

                if get_attr_nodes[node.target] != node:
                    node.replace_all_uses_with(get_attr_nodes[node.target])
                    traced.graph.erase_node(node)

        # avoid looking at next node by keeping track of previous pipe_split
        prev_pipe_split_idx = -1
        pipe_split_nodes_to_erase = set()
        for i, node in enumerate(traced.graph.nodes):
            if (node.op, node.target) == ("call_function", pipe_split):
                if prev_pipe_split_idx == i - 1:
                    pipe_split_nodes_to_erase.add(node)
                prev_pipe_split_idx = i

        for node in pipe_split_nodes_to_erase:
            traced.graph.erase_node(node)

        traced.recompile()

        part_idx = 0

        def split_callback(n: pippy.fx.Node):
            nonlocal part_idx
            if (n.op, n.target) == ("call_function", pipe_split):
                part_idx += 1
            return part_idx

        # Ask split_module to return mapping from new qualname to old qualname
        qualname_map: Dict[str, str] = {}
        # TODO: what does split do with module invocations? does it move the modules
        # into the submodules?
        split = split_module(traced, mod, split_callback, qualname_map)
        # a (custom) tracer can produce dead code like orphan get_attr nodes
        split.graph.eliminate_dead_code()

        # peephole to remove pipe_split
        for submodule in split.modules():
            if isinstance(submodule, pippy.fx.GraphModule):
                for node in submodule.graph.nodes:
                    if (node.op, node.target) == ("call_function", pipe_split):
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

        def move_param_to_callee(
            root, callee_name, param_val, use_idx, is_buffer
        ):
            assert isinstance(param_val, torch.Tensor), (
                f"Expected '{node.target}' to be {torch.Tensor} but got {type(param_val)}."
                + (
                    f" It might happen if module '{node.target}' was passed to some 'leaf function'"
                    f"(see https://pytorch.org/docs/stable/fx.html#pippy.fx.wrap). Please inspect "
                    f"usages of '{node.target}' in the traced graph."
                    if isinstance(param_val, torch.nn.Module)
                    else ""
                )
            )
            callee = root.get_submodule(callee_name)
            new_param_name = f"moved_{node.target.replace('.', '_')}"
            assert not hasattr(
                callee, new_param_name
            ), f"Module {callee_name} already has a parameter named {new_param_name}"
            if is_buffer:
                callee.register_buffer(new_param_name, param_val)
            else:
                setattr(callee, new_param_name, param_val)

            # Update qualname mapping
            # New qualname will have submodule prefix
            new_qualname = f"{callee_name}.{new_param_name}"
            if node.target in qualname_map:
                # Just in case the target name is already in the qualname_map
                # returned by split_module() -- we update the mapping using the
                # new name as a new key
                qualname_map[new_qualname] = qualname_map.pop(node.target)
            else:
                qualname_map[new_qualname] = node.target

            ph_counter = 0
            for sn in callee.graph.nodes:
                if sn.op == "placeholder":
                    if ph_counter == use_idx:
                        with callee.graph.inserting_before(sn):
                            get_attr = callee.graph.get_attr(new_param_name)
                            sn.replace_all_uses_with(get_attr)
                            callee.graph.erase_node(sn)
                    ph_counter += 1
            callee.graph.lint()
            callee.recompile()

            return get_attr

        to_delete = list()  # a list of nodes for deferral deletion

        for node in split.graph.nodes:
            if node.op == "get_attr" and len(node.users) == 1:
                user = list(node.users)[0]
                assert user.op == "call_module"
                use_idx = delete_user_reference(node, user)

                # Move parameter into submodule and replace PH with a get_attr
                atoms = node.target.split(".")
                mod_itr = split
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)
                param_val = getattr(mod_itr, atoms[-1])
                is_buffer = atoms[-1] in mod_itr._buffers

                move_param_to_callee(
                    split, user.target, param_val, use_idx, is_buffer
                )

                to_delete.append((mod_itr, atoms))

        # deferral deletion
        for (mod_itr, atoms) in to_delete:
            delattr(mod_itr, atoms[-1])

        split.graph.lint()
        split.recompile()

        # Handle multi-use parameters based on user's configuration
        # TODO: generalize this to sequential
        multi_use_param_spec = multi_use_param_spec or {}

        multi_use_params_qualnames: Dict[
            str, Optional[MultiUseParameterConfig]
        ] = {}
        for node in split.graph.nodes:
            if node.op == "get_attr" and len(node.users) > 1:
                multi_use_params_qualnames.setdefault(node.target)

        for param in multi_use_params_qualnames:
            if isinstance(multi_use_param_spec, MultiUseParameterConfig):
                multi_use_params_qualnames[param] = multi_use_param_spec
            elif isinstance(multi_use_param_spec, dict):
                multi_use_params_qualnames[param] = multi_use_param_spec.get(
                    param, MultiUseParameterConfig.TRANSMIT
                )
            else:
                raise ValueError(
                    "multi_use_param_spec must be MultiUseParamSpec enum or dict"
                )

        # TODO: do we maintain the invariant that `Node.users` is topologically ordered? I don't think so
        node_to_first_user: Dict[pippy.fx.Node, pippy.fx.Node] = {}
        for node in split.graph.nodes:
            for input in node.all_input_nodes:
                if input not in node_to_first_user:
                    node_to_first_user[input] = node

        for node in split.graph.nodes:
            if (
                node.op == "get_attr"
                and node.target in multi_use_params_qualnames
            ):
                reuse_type = multi_use_params_qualnames[node.target]
                if reuse_type == MultiUseParameterConfig.TRANSMIT:
                    first_user = node_to_first_user[node]
                    assert first_user.op == "call_module"

                    use_idx = delete_user_reference(
                        node, first_user, delete_node=False
                    )

                    atoms = node.target.split(".")
                    mod_itr = split
                    for atom in atoms[:-1]:
                        mod_itr = getattr(mod_itr, atom)
                    param_val = getattr(mod_itr, atoms[-1])
                    is_buffer = atoms[-1] in mod_itr._buffers

                    callee_param_def = move_param_to_callee(
                        split, first_user.target, param_val, use_idx, is_buffer
                    )

                    delattr(mod_itr, atoms[-1])

                    # Add extra output to the callee and switch references to the parameter
                    # access in the pipeline graph to use this.
                    submod = split.get_submodule(first_user.target)
                    callee_output_nodes = [
                        n for n in submod.graph.nodes if n.op == "output"
                    ]
                    assert len(callee_output_nodes) == 1
                    callee_output_node = callee_output_nodes[0]

                    # TODO: zero outputs?
                    if isinstance(callee_output_node.args[0], tuple):
                        new_output_args = callee_output_node.args[0] + (
                            callee_param_def,
                        )
                        callee_output_node.args = (new_output_args,)
                        new_output_idx = len(new_output_args) - 1
                        promoted_to_tuple = False
                    else:
                        new_output_args = (
                            callee_output_node.args[0],
                            callee_param_def,
                        )
                        callee_output_node.args = (new_output_args,)
                        new_output_idx = len(new_output_args) - 1
                        promoted_to_tuple = True

                    submod.graph.lint()
                    submod.recompile()

                    with split.graph.inserting_after(first_user):
                        if promoted_to_tuple:
                            # TODO: test this code path
                            orig_output_getitem = split.graph.call_function(
                                operator.getitem, (first_user, 0)
                            )
                            first_user.replace_all_uses_with(
                                orig_output_getitem
                            )
                            # HACK because the above replace_all_uses with ALSO replaced the instance
                            # of first_user within the getitem node we just added
                            orig_output_getitem.args = (
                                first_user,
                            ) + orig_output_getitem.args[1:]

                        transmitted_value_getitem = split.graph.call_function(
                            operator.getitem, (first_user, new_output_idx)
                        )
                        node.replace_all_uses_with(transmitted_value_getitem)
                        split.graph.erase_node(node)
                elif reuse_type == MultiUseParameterConfig.REPLICATE:
                    for user in copy.copy(node.users):
                        use_idx = delete_user_reference(
                            node, user, delete_node=False
                        )
                        atoms = node.target.split(".")
                        mod_itr = split
                        for atom in atoms[:-1]:
                            mod_itr = getattr(mod_itr, atom)
                        param_val = getattr(mod_itr, atoms[-1])
                        is_buffer = atoms[-1] in mod_itr._buffers

                        move_param_to_callee(
                            split, user.target, param_val, use_idx, is_buffer
                        )

                    atoms = node.target.split(".")
                    mod_itr = split
                    for atom in atoms[:-1]:
                        mod_itr = getattr(mod_itr, atom)

                    delattr(mod_itr, atoms[-1])

                    split.graph.erase_node(node)
                else:
                    raise ValueError(
                        f"Unknown multi-use config value {reuse_type} specified for {node.target}"
                    )

        split.delete_all_unused_submodules()

        split.graph.lint()
        split.recompile()

        num_stages = Pipe._number_and_count_forward_stages(split)

        if isinstance(mod, LossWrapper) or output_loss_value_spec:
            _insert_stage_symbolic_backward(split.graph, output_loss_value_spec)
            split.recompile()
            has_loss_and_backward = True
        else:
            has_loss_and_backward = False

        return Pipe(split, qualname_map, num_stages, has_loss_and_backward)

    @staticmethod
    def from_tracing(
        mod: torch.nn.Module,
        multi_use_param_spec: Optional[MultiUseParamSpec] = None,
        tracer=None,
        output_loss_value_spec=None,
        deep_copy_module=False,
        split_policy: Optional[
            Callable[[pippy.fx.GraphModule], pippy.fx.GraphModule]
        ] = None,
        **kwargs,
    ):
        # TODO: abstract partitioning policy

        global _pipeline_tracer
        old__pipeline_tracer = _pipeline_tracer
        _pipeline_tracer = tracer or pippy.fx.Tracer()
        try:
            # TODO: tracing policy
            if deep_copy_module:
                mod = copy.deepcopy(
                    mod
                )  # because further pipe building activities can modify mod
            graph = _pipeline_tracer.trace(mod, **kwargs)
            if isinstance(graph, torch_fx.Graph):
                # HACK to convert torch.fx.Graph to pippy.fx.Graph
                g_new = pippy.fx.Graph()
                val_map: Dict[pippy.fx.Node, pippy.fx.Node] = {}
                out = g_new.graph_copy(graph, val_map, False)
                g_new.output(out)

                # `pippy.fx.map_arg` doesn't work on torch.fx.Node instances;
                # do it here
                def remap_vals(n):
                    return val_map[n]

                for node in g_new.nodes:
                    node.args = torch_fx.map_arg(node.args, remap_vals)
                    node.kwargs = torch_fx.map_arg(node.kwargs, remap_vals)
                graph = g_new

            traced = pippy.fx.GraphModule(mod, graph)
        finally:
            _pipeline_tracer = old__pipeline_tracer

        if split_policy is not None:
            traced = split_policy(traced)

        return Pipe._from_traced(
            mod,
            traced,
            multi_use_param_spec,
            output_loss_value_spec=output_loss_value_spec,
        )

    def __str__(self):
        return self.split_gm.__str__()

    def __repr__(self):
        return self.split_gm.__repr__()

    def defer_stage_init(self, device):
        def materialize_stage(target: str) -> torch.nn.Module:
            logging.info(f"Materializing {target} on {device}")
            return self.split_gm.get_submodule(target).to(device)

        setattr(Pipe, "materialize_stage", materialize_stage)

    def distx_materialize_stage(self, stage: int, device):
        logging.info(f"Materializing stage {stage} on {device}")
        split_gm_children = list(self.split_gm.children())
        submod = split_gm_children[stage]
        deferred_init.materialize_module(submod)
        #print(f"Stage {stage} is on {_find_mod_device(submod)} after materialization")
        self.submod = submod

        def materialize_stage(target: str) -> torch.nn.Module:
            logging.info(f"Returning {target}")
            return self.submod

        setattr(Pipe, "materialize_stage", materialize_stage)
        return submod

    @staticmethod
    def is_stage_init_deferred():
        return hasattr(Pipe, "materialize_stage")


class PipeSplitWrapper(torch.nn.Module):
    class SplitPoint(Enum):
        BEGINNING = 1
        END = 2

    def __init__(
        self,
        mod: torch.nn.Module,
        split_point: SplitPoint = SplitPoint.BEGINNING,
    ):
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


def annotate_split_points(
    mod: torch.nn.Module, spec: Dict[str, PipeSplitWrapper.SplitPoint]
):
    # TODO: make this implementation out-of-place?
    for qualname, split_type in spec.items():
        atoms = qualname.split(".")
        predecessor_module = mod
        for i, atom in enumerate(atoms[:-1]):
            try:
                predecessor_module = getattr(predecessor_module, atom)
            except AttributeError as e:
                raise AttributeError(
                    f'Specified target {qualname} referenced nonexistent module {".".join(atoms[:i+1])}'
                )

        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        wrapped_mod = PipeSplitWrapper(mod_to_wrap, split_type)
        setattr(predecessor_module, atoms[-1], wrapped_mod)
