import logging
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import cast, Dict, List, Optional, Sequence, Set, Tuple

import functorch.compile

import torch
import torch.fx as fx
import torch.nn as nn
from functorch.compile import aot_module, make_boxed_func
from spmd.tensor import DeviceMesh, distribute_tensor, DTensor
from spmd.tensor.dispatch import (
    _CURRENT_DECOMPOSITION_TABLE,
    operator_dispatch,
    propagate_input_sharding,
)
from spmd.tensor.placement_types import _Partial, Placement, Replicate, Shard
from spmd.tensor.redistribute import _redistribute_with_local_tensor
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .aot_function_patch import patched_aot_function
from .graph_utils import OP
from .log_utils import rank0_info


# patch aot_function so that we can pass the full (non-sharded) input to capture the graph
# pyre-fixme
functorch._src.aot_autograd.aot_function = patched_aot_function


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
) -> fx.GraphModule:
    """
    creates a graph for a dummy add function from a partial DTensor.
    This dummy add is used for triggering all_reduce on a Partial DTensor
    during the DTensor expansion of the traced graph.
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

    return traced_dispatch


def _convert_output(
    gm: fx.GraphModule,
    node: fx.Node,
    node_to_obj: Dict[fx.Node, object],
) -> None:
    new_args = []
    has_partial = False
    for argument in node.args[0]:  # type: ignore
        if not isinstance(argument, fx.Node):
            new_args.append(argument)
            continue
        obj = node_to_obj[argument]

        has_partial = _is_partial_dtensor(obj)

        if not has_partial:
            continue

        # we know it's a dtensor from is partial DT check...
        dt = cast(DTensor, obj)

        traced_dispatch = _build_dummy_add_graph(dt, node_to_obj)

        wait = [n for n in traced_dispatch.graph.nodes if n.name == "wait_comm"]
        add = [n for n in traced_dispatch.graph.nodes if n.name == "add"]
        assert len(wait) == 1 and len(add) == 1
        add[0].replace_all_uses_with(wait[0])
        traced_dispatch.graph.lint()
        traced_dispatch.graph.eliminate_dead_code()

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
        gm.graph.output(new_args)
    else:
        rank0_info(logger, "The output does not have partial arguments.")
    return


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


def _convert_to_distributed_graph(
    training_phase: TrainingPhase,
    gm: fx.GraphModule,
    inputs: List[torch.Tensor],
    schemas: List[Schema],
    _allow_partial: bool = False,
) -> fx.GraphModule:
    node_to_obj: Dict[fx.Node, object] = {}
    # map local op node in traced_f to its corresponding subgraph of
    # DTensor ops.
    node_replacements: Dict[torch.fx.Node, torch.fx.GraphModule] = {}

    rank0_info(logger, f"Training phase: {training_phase}")
    for i, node in enumerate(gm.graph.nodes):
        rank0_info(logger, f"node{i}: op={node.op} target={node.target}")
        if node.op == OP.PLACEHOLDER:
            assert i < len(
                inputs
            ), f"got more placeholer nodes ({i + 1}) than inputs ({len(inputs)})"
            if training_phase == TrainingPhase.FORWARD:
                # in the forward phase we start with the full "global" ensors.
                # we needed this because we needed to capture the original graph.
                node_to_obj[node] = distribute_tensor(  # DTensor.from_local(
                    inputs[i],
                    schemas[i].mesh,
                    schemas[i].placements,
                )
            else:
                # But in the backward pass we got "real" sharded inputs
                # so we have to actually make DTensors out of them
                assert training_phase == TrainingPhase.BACKWARD
                node_to_obj[node] = DTensor.from_local(
                    inputs[i],
                    schemas[i].mesh,
                    schemas[i].placements,
                )

        elif isinstance(node.target, torch._ops.OpOverload):
            node_replacements[node] = _get_dtensor_dispatch_graph(
                node, node_to_obj
            )
        elif node.op == OP.OUTPUT:
            if not _allow_partial:
                _convert_output(gm, node, node_to_obj)
                break
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
    return gm


class _SPMD:
    def __init__(self, orig_module: nn.Module, param_schema: Schema) -> None:
        self._orig_module = orig_module
        self._param_schema = param_schema
        self._fwd_gm: Optional[fx.GraphModule] = None
        self._bwd_gm: Optional[fx.GraphModule] = None

    def _compile(
        self,
        training_phase: TrainingPhase,
        gm: fx.GraphModule,
        inputs: List[torch.Tensor],
    ) -> fx.GraphModule:
        def is_param(t: torch.Tensor) -> bool:
            # N.B.: id(t) and id(param) does not match
            return t.storage().data_ptr() in (
                p.storage().data_ptr() for p in self._orig_module.parameters()
            )

        shard_schema: Schema = Schema(
            mesh=self._param_schema.mesh, placements=[Shard(0)]
        )
        schemas: List[Schema] = [
            self._param_schema if is_param(inp) else shard_schema
            for inp in inputs
        ]

        gm = _convert_to_distributed_graph(
            training_phase,
            gm,
            inputs,
            schemas,
        )
        if training_phase == TrainingPhase.FORWARD:
            self._fwd_gm = gm
        elif training_phase == TrainingPhase.BACKWARD:
            self._bwd_gm = gm
        return make_boxed_func(gm)


def distribute(
    module: nn.Module,
    param_schema: Schema,
    fw_only: bool,
    *args: Tuple[object],
    **kwargs: Dict[str, object],
) -> Tuple[nn.Module, fx.GraphModule, Optional[fx.GraphModule]]:
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

    spmd = _SPMD(module, param_schema)
    compiled_m = aot_module(
        module,
        partial(spmd._compile, TrainingPhase.FORWARD),
        partial(spmd._compile, TrainingPhase.BACKWARD),
        pre_compile_fn=gather_inputs_for_compilation,
        decompositions=_CURRENT_DECOMPOSITION_TABLE,
    )

    # Forcing to execute one step to get the forward and backwarg graph.
    # One major issue of this flow is the optimizer states. All optimizer
    # states are lazily initialized by default but PT-D distributed checkpoint
    # requires eagerly initialization states. Eager initialization means the
    # optimizer state is initialized and loaded before ``forward()`` is called.
    # One way to solve the issue is to ensure that this ``distribute()`` is
    # called before optimizer ``load_state_dict()`` is executed.
    output = cast(nn.Module, compiled_m)(*args, **kwargs)
    if not fw_only:
        # fw_only is a hack if we cannot compile the backward. Should be a
        # temporary solution.
        # Force to compile the backward
        output.sum().backward()
        # Clear the gradient
        for p in compiled_m.parameters():
            if p.grad is not None:
                p.grad = None

    assert spmd._fwd_gm is not None
    return compiled_m, spmd._fwd_gm, spmd._bwd_gm
