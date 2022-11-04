import logging

import torch
import torch.fx as fx
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils._pytree import tree_flatten, tree_map

from functorch.compile import aot_module

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, cast

from spmd.tensor import (
    DTensor,
    DeviceMesh,
)
from spmd.tensor.placement_types import Placement, Shard, Replicate, _Partial
from spmd.tensor.dispatch import operator_dispatch, propagate_input_sharding
from spmd.tensor.redistribute import _redistribute_with_local_tensor

from . import config

log = logging.getLogger(__name__)

@dataclass
class Schema:
    mesh: DeviceMesh
    placements: List[Placement]


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
        if isinstance(arg, DTensor) and redistribute:
            specs[arg._local_tensor] = (
                arg.size(),
                flatten_args_schema[i].mesh,
                arg.placements,
                flatten_args_schema[i].placements,
            )

    dispatch = partial(
        _dispatch_with_local_tensors,
        op_overload,
        kwargs=kwargs,
        specs=specs,
    )

    def unwrap_local(e: object) -> object:
        return e._local_tensor if isinstance(e, DTensor) else e

    return make_fx(dispatch)(tree_map(unwrap_local, args))


def _convert_to_distributed(
    gm: fx.GraphModule,
    inps: List[torch.Tensor],
    schemas: List[Schema],
    _allow_partial: bool = False,
    debug_str="",
) -> fx.GraphModule:
    node_to_obj: Dict[fx.Node, object] = {}
    # map local op node in traced_f to its corresponding subgraph of
    # DTensor ops.
    replacements: Dict[torch.fx.Node, torch.fx.GraphModule] = {}
    log.debug(f"{debug_str} Original graph {gm}")
    for i, node in enumerate(gm.graph.nodes):
        if node.op == "placeholder":
            assert i < len(
                inps
            ), f"got more placeholer nodes ({i + 1}) than inputs ({len(inps)})"
            node_to_obj[node] = DTensor.from_local(
                inps[i], schemas[i].mesh, schemas[i].placements
            )
        elif isinstance(node.target, torch._ops.OpOverload):
            replacements[node] = _get_dtensor_dispatch_graph(node, node_to_obj)
        elif node.op == "output":
            if not _allow_partial:
                new_args = []
                has_partial = False
                for a in node.args[0]:
                    if not isinstance(a, fx.Node):
                        new_args.append(a)
                        continue
                    obj = node_to_obj[a]
                    if isinstance(obj, DTensor) and isinstance(
                        obj.placements[0], _Partial
                    ):
                        has_partial = True

                        def dummy_add(
                            grad: torch.Tensor, zero: torch.Tensor
                        ) -> torch.Tensor:
                            return grad + zero

                        grad: torch.Tensor = obj._local_tensor
                        zero: torch.Tensor = torch.zeros_like(obj._local_tensor)

                        traced_add = make_fx(dummy_add)(grad, zero)

                        placeholders = [
                            n
                            for n in traced_add.graph.nodes
                            if n.op == "placeholder"
                        ]
                        call_functions = [
                            n
                            for n in traced_add.graph.nodes
                            if n.op == "call_function"
                        ]
                        assert len(placeholders) == 2
                        assert len(call_functions) == 1
                        node_to_obj[placeholders[0]] = obj
                        node_to_obj[placeholders[1]] = zero
                        traced_dispatch = _get_dtensor_dispatch_graph(
                            call_functions[0], node_to_obj
                        )
                        traced_dispatch.graph.lint()

                        wait = [
                            n
                            for n in traced_dispatch.graph.nodes
                            if n.name == "wait_comm"
                        ]
                        add = [
                            n
                            for n in traced_dispatch.graph.nodes
                            if n.name == "add"
                        ]
                        assert len(wait) == 1 and len(add) == 1
                        add[0].replace_all_uses_with(wait[0])
                        traced_dispatch.graph.lint()
                        traced_dispatch.graph.eliminate_dead_code()

                        value_remap: Dict[fx.Node, fx.Node] = {}
                        for dtn in traced_dispatch.graph.nodes:
                            if dtn.op == "placeholder":
                                # do nothing, ignore placeholders, as it has
                                # already been prepared in value_remap
                                value_remap[dtn] = a
                            elif dtn.op == "output":
                                assert (
                                    len(dtn.args) == 1 and len(dtn.args[0]) == 1
                                ), f"Expecting single output, but got {dtn.args}"
                                new_args.append(value_remap[dtn.args[0][0]])
                            else:
                                if dtn.op == "get_attr":
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
                    gm.graph.erase_node(node)
                    gm.graph.output(new_args)
                break
        else:
            raise ValueError(f"Unrecognized node {node}")

    log.debug(f"{debug_str} graph post allreduce insertion {gm}")

    # replace nodes in local traced graph with DTensor's dispatch graph
    for node in gm.graph.nodes:
        if node not in replacements:
            continue

        traced_dispatch = replacements[node]
        # Map DT's dispatch graph input placeholder nodes to the ones in
        # local traced graph. It uses index-based accessing, which is
        # brittle, just for testing purpose.
        flatten_args, _ = tree_flatten(node.args)
        i, value_remap = 0, {}
        for dtn in traced_dispatch.graph.nodes:
            if dtn.op == "placeholder":
                value_remap[dtn] = flatten_args[i]
                i += 1

        # insert DT's dispatch graph to traced local graph.
        with gm.graph.inserting_before(node):
            for dtn in traced_dispatch.graph.nodes:
                if dtn.op == "placeholder":
                    # do nothing, ignore placeholders, as it has already
                    # been prepared in value_remap
                    pass
                elif dtn.op == "output":
                    assert (
                        len(dtn.args) == 1 and len(dtn.args[0]) == 1
                    ), f"Expecting single output, but got {dtn.args}"
                    node.replace_all_uses_with(value_remap[dtn.args[0][0]])
                else:
                    value_remap[dtn] = gm.graph.node_copy(
                        dtn, lambda n: value_remap[n]
                    )

    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    log.debug(f"{debug_str} graph after dtensor subgraph {gm}")
    return gm


class SPMD(nn.Module):
    # TODO: add schema_override
    def __init__(self, module: nn.Module, schema: Schema) -> None:
        super().__init__()
        init_logging(config.log_level)
        assert schema.placements == [
            Replicate()
        ], "SPMD only support Replicate() parameters for now"

        # TODO: coalesce broadcasts
        for p in module.parameters():
            dist.broadcast(p, src=0)

        self._local_module: nn.Module = module
        self._schema: Schema = schema
        self._compiled_m: Optional[nn.Module] = None

    def _fw_compile(self, gm: fx.GraphModule, inps: List[torch.Tensor]
    ) -> fx.GraphModule:
        def is_param(t: torch.Tensor) -> bool:
            # N.B.: id(t) and id(param) does not match
            return t.storage().data_ptr() in [
                p.storage().data_ptr() for p in self._local_module.parameters()
            ]

        shard_schema: Schema = Schema(
            mesh=self._schema.mesh, placements=[Shard(0)]
        )
        schemas: List[Schema] = [
            self._schema if is_param(inp) else shard_schema for inp in inps
        ]
        log.debug(f"schemas for FW pass are {schemas} with num inputs {len(inps)}")
        return _convert_to_distributed(gm, inps, schemas, debug_str="fw")

    def _bw_compile(
        self, gm: fx.GraphModule, inps: List[torch.Tensor]
    ) -> fx.GraphModule:
        def is_param(t: torch.Tensor) -> bool:
            # N.B.: id(t) and id(param) does not match
            return t.storage().data_ptr() in [
                p.storage().data_ptr() for p in self._local_module.parameters()
            ]

        shard_schema: Schema = Schema(
            mesh=self._schema.mesh, placements=[Shard(0)]
        )
        schemas: List[Schema] = [
            self._schema if is_param(inp) else shard_schema for inp in inps
        ]
        log.debug(f"schemas for BW pass are {schemas} with num inputs {len(inps)}")
        return _convert_to_distributed(gm, inps, schemas, debug_str="bw")

    def forward(
        self, *args: Tuple[object], **kwargs: Dict[str, object]
    ) -> object:
        if self._compiled_m is None:
            self._compiled_m = aot_module(
                self._local_module,
                self._fw_compile,
                self._bw_compile,
            )

        return cast(nn.Module, self._compiled_m)(*args, **kwargs)
