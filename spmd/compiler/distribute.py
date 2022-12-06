import logging
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import cast, Dict, List, Sequence, Set, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from functorch.compile import aot_module, make_boxed_func

from spmd.tensor import (
    _CURRENT_DECOMPOSITION_TABLE,
    _Partial,
    _redistribute_with_local_tensor,
    DeviceMesh,
    distribute_tensor,
    DTensor,
    operator_dispatch,
    Placement,
    propagate_input_sharding,
    Replicate,
    Shard,
)
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .aot_function_patch import patched_aot_function
from .distributed_graph import DistributedGraph
from .graph_utils import CommType, get_comm_block_ops, OP
from .log_utils import rank0_info

# patch aot_function so that we can pass the full (non-sharded) input to capture the graph
# pyre-fixme
torch._functorch.aot_autograd.aot_function = patched_aot_function


logger: logging.Logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass
class Schema:
    mesh: DeviceMesh
    placements: List[Placement]


def _is_partial_dtensor(obj: object) -> bool:
    """check if object is 1) DTensor and  2) with any placement of _Partial"""
    if not isinstance(obj, DTensor):
        return False

    is_partial = False
    for placement in obj.placements:
        if isinstance(placement, _Partial):
            is_partial = True
            break

    return is_partial


def _dispatch_with_local_tensors(
    op: torch._ops.OpOverload,
    local_args: Tuple[object, ...],
    kwargs: Dict[str, object] = {},
    specs: Dict[
        torch.Tensor,
        Tuple[torch.Size, DeviceMesh, Sequence[Placement], Sequence[Placement]],
    ] = {},
) -> object:
    def redistribute(arg: object) -> object:
        return (
            _redistribute_with_local_tensor(arg, *specs[arg])
            if isinstance(arg, torch.Tensor) and arg in specs
            else arg
        )

    # FIXME: this is broken because it won't redistributed potential tensors on the kwargs
    return op(*tree_map(redistribute, local_args), **kwargs)


def _get_dtensor_dispatch_graph(
    node: fx.Node,
    node_to_obj: Dict[fx.Node, object],
) -> fx.GraphModule:
    def remap_arg(arg: object) -> object:
        if isinstance(arg, torch.fx.Node):
            obj = node_to_obj[arg]
            if _get_tracer(obj):
                # This is a shared arg, already has a tracer from previous
                # tracing. Delete the tracer.
                del cast(Dict[object, object], obj.__dict__)[proxy_slot]
            return obj
        else:
            return arg

    args = tree_map(remap_arg, node.args)
    # kwargs in this set of tests are all constants
    kwargs = cast(Dict[str, object], node.kwargs)

    op_overload = cast(torch._ops.OpOverload, node.target)

    # run dispatch once to get the real DTensor output
    out = operator_dispatch(
        op_overload,
        args,
        kwargs,  # kwargs in this set of tests are all constants
        DTensor._op_to_rules,
        DTensor._custom_dispatch_ops,
    )
    node_to_obj[node] = out

    # get DTensor specs for inputs and outputs
    (target_schema, redistribute, output_sharding,) = propagate_input_sharding(
        op_overload,
        args,
        kwargs,
        DTensor._op_to_rules,
    )
    # ===== Begin code taken from pack_args_kwargs_with_local_tensor =====
    flatten_args, args_tree_spec = tree_flatten(args)
    flatten_args_schema, _ = tree_flatten(target_schema.args_schema)

    specs: Dict[
        torch.Tensor,
        Tuple[
            torch.Size,
            DeviceMesh,
            Sequence[Placement],
            Sequence[Placement],
        ],
    ] = {}
    for i, arg in enumerate(flatten_args):
        if isinstance(arg, DTensor):
            if redistribute:
                specs[arg._local_tensor] = (
                    arg.size(),
                    flatten_args_schema[i].mesh,
                    arg.placements,
                    flatten_args_schema[i].placements,
                )
            flatten_args_schema[i] = arg._local_tensor

    local_target_args = tree_unflatten(flatten_args_schema, args_tree_spec)
    # ===== End code taken from pack_args_kwargs_with_local_tensor =====

    # FIXME: this is broken when kwargs contains tensors
    #        or if a non-tensor kwarg was modified by the sharding propagation
    #        (in order to fix, need to port over pack_args_kwargs_with_local_tensor for kwargs as well)

    dispatch = partial(
        _dispatch_with_local_tensors,
        op_overload,
        kwargs=kwargs,
        specs=specs,
    )

    def unwrap_local(e: object) -> object:
        return e._local_tensor if isinstance(e, DTensor) else e

    return make_fx(dispatch)(local_target_args)


def _build_dummy_add_graph(
    dt: DTensor, node_to_obj: Dict[fx.Node, object]
) -> Tuple[fx.GraphModule, object]:
    """
    creates a graph for a dummy add function from a partial DTensor.
    This dummy add is used for triggering all_reduce on a Partial DTensor
    during the DTensor expansion of the traced graph.
    Also returns the actual DTensor after resharding.
    """

    def dummy_add(grad: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        return grad + zero

    grad: torch.Tensor = dt._local_tensor
    zero: torch.Tensor = torch.zeros_like(dt._local_tensor)

    traced_add = make_fx(dummy_add)(grad, zero)

    placeholders = [n for n in traced_add.graph.nodes if n.op == OP.PLACEHOLDER]
    call_functions = [
        n for n in traced_add.graph.nodes if n.op == OP.CALL_FUNCTION
    ]
    assert len(placeholders) == 2
    assert len(call_functions) == 1
    node_to_obj[placeholders[0]] = dt
    node_to_obj[placeholders[1]] = zero

    traced_dispatch = _get_dtensor_dispatch_graph(
        call_functions[0], node_to_obj
    )

    traced_dispatch.graph.lint()

    return traced_dispatch, node_to_obj[call_functions[0]]


def _convert_output(
    gm: fx.GraphModule,
    node: fx.Node,
    node_to_obj: Dict[fx.Node, object],
) -> fx.Node:
    new_args = []
    has_partial = False
    for argument in node.args[0]:  # type: ignore
        if not isinstance(argument, fx.Node):
            new_args.append(argument)
            continue

        obj = node_to_obj[argument]

        if not _is_partial_dtensor(obj):
            new_args.append(argument)
            continue

        has_partial = True

        # we know it's a dtensor from is partial DT check...
        dt = cast(DTensor, obj)

        traced_dispatch, result_obj = _build_dummy_add_graph(dt, node_to_obj)

        wait = [n for n in traced_dispatch.graph.nodes if n.name == "wait_comm"]
        add = [n for n in traced_dispatch.graph.nodes if n.name == "add"]
        assert len(wait) == 1 and len(add) == 1

        # remove add node and replace it with wait node
        add[0].replace_all_uses_with(wait[0])
        traced_dispatch.graph.lint()
        traced_dispatch.graph.eliminate_dead_code()
        # also update the actual DTensor corresponding to the node
        node_to_obj[wait[0]] = result_obj

        value_remap: Dict[fx.Node, fx.Node] = {}
        for dtn in traced_dispatch.graph.nodes:
            if dtn.op == OP.PLACEHOLDER:
                # do nothing, ignore placeholders, as it has
                # already been prepared in value_remap
                value_remap[dtn] = argument
            elif dtn.op == OP.OUTPUT:
                assert (
                    len(dtn.args) == 1 and len(dtn.args[0]) == 1
                ), f"Expecting single output, but got {dtn.args} {len(dtn.args)}"
                new_args.append(value_remap[dtn.args[0][0]])
                # the concrete DTensor value of output was added when creating the
                # inner graph (in _build_dummy_add_graph). Just add it to the final
                # output node so that we can report the final output specs correctly.
                node_to_obj[value_remap[dtn.args[0][0]]] = node_to_obj[
                    dtn.args[0][0]
                ]
            else:
                if dtn.op == OP.GET_ATTR:
                    setattr(
                        gm,
                        dtn.target,
                        getattr(traced_dispatch, dtn.target),
                    )
                with gm.graph.inserting_before(node):
                    value_remap[dtn] = gm.graph.node_copy(
                        dtn, lambda n: value_remap[n]
                    )
    if has_partial:
        rank0_info(logger, "The output has partial arguments.")
        gm.graph.erase_node(node)
        return gm.graph.output(new_args)
    else:
        rank0_info(logger, "The output does not have partial arguments.")
        return node


def _rebuild_graph(
    gm: fx.GraphModule,
    node_replacements: Dict[torch.fx.Node, torch.fx.GraphModule],
) -> None:
    # replace nodes in local traced graph with DTensor's dispatch graph
    for node in gm.graph.nodes:
        if node not in node_replacements:
            continue

        traced_dispatch = node_replacements[node]
        # Map DT's dispatch graph input placeholder nodes to the ones in
        # local traced graph. It uses index-based accessing, which is
        # brittle, just for testing purpose.
        flatten_args, _ = tree_flatten(node.args)
        i, value_remap = 0, {}
        for dtn in traced_dispatch.graph.nodes:
            if dtn.op == OP.PLACEHOLDER:
                value_remap[dtn] = flatten_args[i]
                i += 1

        # insert DT's dispatch graph to traced local graph.
        with gm.graph.inserting_before(node):
            for dtn in traced_dispatch.graph.nodes:
                if dtn.op == OP.PLACEHOLDER:
                    # do nothing, ignore placeholders, as it has already
                    # been prepared in value_remap
                    pass
                elif dtn.op == OP.OUTPUT:
                    # TODO: AssertionError: Expecting single output, but got ([getitem, getitem_1, getitem_2],)
                    assert (
                        len(dtn.args) == 1
                    ), f"Expecting single output, but got {dtn.args} {len(dtn.args[0])}"
                    node.replace_all_uses_with(value_remap[dtn.args[0][0]])
                else:
                    value_remap[dtn] = gm.graph.node_copy(
                        dtn, lambda n: value_remap[n]
                    )

    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()


def _convert_to_distributed(
    training_phase: TrainingPhase,
    gm: fx.GraphModule,
    inps: List[torch.Tensor],
    schemas: List[Schema],
    _allow_partial: bool = False,
) -> Tuple[fx.GraphModule, Dict[str, Schema]]:
    """
    Returns:
        - transformed graph module
        - map from output name to DTensorSpec
    """
    node_to_obj: Dict[fx.Node, object] = {}
    # map local op node in traced_f to its corresponding subgraph of
    # DTensor ops.
    node_replacements: Dict[torch.fx.Node, torch.fx.GraphModule] = {}

    rank0_info(logger, f"Training phase: {training_phase}")
    output_schemas: Dict[str, Schema] = {}
    for i, node in enumerate(gm.graph.nodes):
        rank0_info(logger, f"node{i}: op={node.op} target={node.target}")
        if node.op == OP.PLACEHOLDER:
            assert i < len(
                inps
            ), f"got more placeholer nodes ({i + 1}) than inputs ({len(inps)})"
            if training_phase == TrainingPhase.FORWARD:
                # in the forward phase we start with the full "global" ensors.
                # we needed this because we needed to capture the original graph.
                node_to_obj[node] = distribute_tensor(
                    inps[i],
                    schemas[i].mesh,
                    schemas[i].placements,
                )
            else:
                # But in the backward pass we got "real" sharded inputs
                # so we have to actually make DTensors out of them
                assert training_phase == TrainingPhase.BACKWARD
                node_to_obj[node] = DTensor.from_local(
                    inps[i],
                    schemas[i].mesh,
                    schemas[i].placements,
                    # prevent running this collective in backwards pass
                    run_check=False,
                )

        elif isinstance(node.target, torch._ops.OpOverload):
            node_replacements[node] = _get_dtensor_dispatch_graph(
                node, node_to_obj
            )
        elif node.op == OP.OUTPUT:
            if not _allow_partial:
                # returns the possibly modified output node
                node = _convert_output(gm, node, node_to_obj)

            # save output sharding for the inputs to backward pass
            for a in node.args[0]:
                if isinstance(a, fx.Node):
                    obj = node_to_obj[a]
                    if isinstance(obj, DTensor):
                        output_schemas[a.name] = Schema(
                            obj.device_mesh, obj.placements  # type: ignore
                        )

        elif node.op == OP.CALL_FUNCTION:

            def remap_arg(arg: object) -> object:
                if isinstance(arg, torch.fx.Node):
                    obj = node_to_obj[arg]
                    # TODO(anj): we need this for getitem but can we be more generic?
                    if isinstance(obj, tuple):
                        return obj
                    if _get_tracer(obj):
                        # This is a shared arg, already has a tracer from previous
                        # tracing. Delete the tracer.
                        del cast(Dict[object, object], obj.__dict__)[proxy_slot]
                    return obj
                else:
                    return arg

            args = tree_map(remap_arg, node.args)
            node_to_obj[node] = node.target(args[0], args[1])
        else:
            raise ValueError(f"Unrecognized node {node}")

    _rebuild_graph(gm, node_replacements)

    return gm, output_schemas


class _SPMD:
    def __init__(
        self,
        dist_graph: DistributedGraph,
        param_schema: Schema,
        input_schemas: Sequence[Placement],
        map_param_and_grad: bool = False,
    ) -> None:
        self._dist_graph = dist_graph
        self._param_schema = param_schema
        # Override the default sharding of input to the model.
        self._input_schemas = input_schemas
        # used to propagate sharding from the output of the forward pass to
        # the input of backward pass
        self._known_specs_by_node_name: Dict[str, Schema] = {}
        # A switch that allow users to turn off map_param_and_grad since it is
        # brittle as for now (the impl depends on the param and grad order).
        self._map_param_and_grad = map_param_and_grad

    def _is_param(self, t: torch.Tensor) -> bool:
        # N.B.: id(t) and id(param) does not match
        orig_module = cast(nn.Module, self._dist_graph.orig_module)
        return t.storage().data_ptr() in (
            p.storage().data_ptr() for p in orig_module.parameters()
        )

    def _map_primal_to_param(
        self, gm: fx.GraphModule, inputs: List[torch.Tensor], nparams: int
    ) -> None:
        self._dist_graph.primal_name_to_node.append({})
        self._dist_graph.primal_to_param.append({})
        for inp, node in zip(inputs, gm.graph.nodes):
            if self._is_param(inp) and node.name.startswith("primals_"):
                self._dist_graph.primal_name_to_node[0][node.name] = node
                self._dist_graph.primal_to_param[0][node] = cast(
                    nn.Parameter, inp
                )
        assert len(self._dist_graph.primal_name_to_node[0]) == nparams, (
            len(self._dist_graph.primal_name_to_node[0]),
            nparams,
        )

    def _map_grad_to_param(self, gm: fx.GraphModule) -> None:
        # This assumes that the order of gradients are in the same order
        # as the parameters in the input. The last argument of the output
        # is None.
        # TODO: this is very brittle, verify how we can do better than this.
        self._dist_graph.grad_to_primal.append({})
        for node in gm.graph.nodes:
            if node.op != OP.OUTPUT:
                continue
            params = list(self._dist_graph.primal_name_to_node[0].values())
            grads = node.args[0][: len(params)]
            assert len(grads) == len(params), (grads, len(params))
            all(grad is not None for grad in grads)
            for grad, param in zip(grads, params):
                if grad.name.startswith("wait_comm"):
                    comm_idx, comm_block_ops = get_comm_block_ops(
                        grad, CommType.ALLREDUCE
                    )
                    comm_node = comm_block_ops[comm_idx]
                    grad = cast(Tuple[fx.Node, ...], comm_node.args[0])[0]
                self._dist_graph.grad_to_primal[0][grad] = param

    def _compile(
        self,
        training_phase: TrainingPhase,
        gm: fx.GraphModule,
        inps: List[torch.Tensor],
    ) -> fx.GraphModule:
        shard_schema: Schema = Schema(
            mesh=self._param_schema.mesh, placements=[Shard(0)]
        )
        schemas: List[Schema] = []
        inp_schema_count = 0
        nparams = 0

        # iterate through inputs (and initial nodes of the graph that should
        # correspond 1:1 to those inputs)
        for inp, placeholder_node in zip(inps, gm.graph.nodes):
            # This is a no-op but we want the order of schemas
            # to match the order of inputs when we iterate through
            # the graph. Usually the non-tensor inputs are at the
            # end of the list so we could drop the schemas for it.

            assert placeholder_node.op == "placeholder", (
                "Expected initial nodes of the GraphModule to be input placeholders. "
                "Got {placeholder_node.op}"
            )

            known_schema = self._known_specs_by_node_name.get(
                placeholder_node.name
            )

            if known_schema is not None:
                schemas.append(known_schema)
            elif not isinstance(inp, torch.Tensor):
                schemas.append(
                    Schema(
                        mesh=self._param_schema.mesh, placements=[Replicate()]
                    )
                )
            else:
                if self._is_param(inp):
                    schemas.append(self._param_schema)
                    nparams += 1
                elif self._input_schemas:
                    schemas.append(self._input_schemas[inp_schema_count])  # type: ignore
                    inp_schema_count += 1
                else:
                    schemas.append(shard_schema)

        parallelized_gm, output_specs = _convert_to_distributed(
            training_phase,
            gm,
            inps,
            schemas,
            _allow_partial=False,
        )
        self._known_specs_by_node_name.update(output_specs)

        if training_phase == TrainingPhase.FORWARD:
            self._dist_graph.fwd_graph_modules.append(parallelized_gm)
            if self._map_param_and_grad:
                self._map_primal_to_param(parallelized_gm, inps, nparams)
        elif training_phase == TrainingPhase.BACKWARD:
            self._dist_graph.bwd_graph_modules.append(parallelized_gm)
            if self._map_param_and_grad:
                self._map_grad_to_param(parallelized_gm)
        return make_boxed_func(parallelized_gm)


def distribute(
    dist_graph: DistributedGraph,
    param_schema: Schema,
    input_schemas: Sequence[Placement],
    force_compile: bool,
    map_param_and_grad: bool,
    *args: Tuple[object],
    **kwargs: Dict[str, object],
) -> nn.Module:
    flat_args, _ = tree_flatten(args)
    flat_kwargs, _ = tree_flatten(kwargs)
    input_set: Set[object] = set(flat_args + flat_kwargs)

    def gather_inputs_for_compilation(
        inps: Tuple[object, ...],
    ) -> Tuple[object, ...]:
        compile_inps = tuple(
            x
            if not isinstance(x, torch.Tensor) or x not in input_set
            else DTensor.from_local(x, param_schema.mesh, [Shard(0)])
            .redistribute(param_schema.mesh, [Replicate()])
            .to_local()
            for x in inps
        )
        return compile_inps

    spmd = _SPMD(dist_graph, param_schema, input_schemas, map_param_and_grad)
    compiled_m = aot_module(
        cast(nn.Module, dist_graph.orig_module),
        partial(spmd._compile, TrainingPhase.FORWARD),
        partial(spmd._compile, TrainingPhase.BACKWARD),
        pre_compile_fn=gather_inputs_for_compilation,
        decompositions=_CURRENT_DECOMPOSITION_TABLE,
    )

    # Force to execute one step to get the forward and backward graph.
    # One major issue of this flow is the optimizer states. All optimizer
    # states are lazily initialized by default but PT-D distributed checkpoint
    # requires eagerly initialization states. Eager initialization means the
    # optimizer state is initialized and loaded before ``forward()`` is called.
    # One way to solve the issue is to ensure that this ``distribute()`` is
    # called before optimizer ``load_state_dict()`` is executed.
    #
    # TODO(chienchin): figure out how to be compatible with optimizer state_dict
    # loading.
    if force_compile:
        output = compiled_m(*args, **kwargs)
        # Force to compile the backward
        output.sum().backward()
        # Clear the gradient
        for p in compiled_m.parameters():
            if p.grad is not None:
                p.grad = None

    return compiled_m
