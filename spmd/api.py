import torch
import torch.fx as fx
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import get_global_rank, get_world_size
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.testing._internal.common_utils import run_tests
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.utils._pytree import tree_flatten, tree_map
from functorch.compile import aot_module
from functorch._src.named_members_polyfill import _named_buffers, _named_parameters

from dataclasses import dataclass

from spmd.tensor import (
    _Partial,
    DTensor,
    DeviceMesh,
    Placement,
    Replicate,
    Shard,
)
from spmd.tensor.dispatch import operator_dispatch, propagate_input_sharding
from spmd.tensor.redistribute import _redistribute_with_local_tensor

import os
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple


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
) -> Any:
    def redistribute(arg):
        return (
            _redistribute_with_local_tensor(arg, *specs[arg])
            if arg in specs
            else arg
        )

    return op(*tree_map(redistribute, local_args), **kwargs)


def _get_dtensor_dispatch_graph(
    node: fx.Node, node_to_obj: Dict[fx.Node, object],
) -> fx.GraphModule:

    def remap_arg(arg):
        if isinstance(arg, torch.fx.Node):
            obj = node_to_obj[arg]
            if _get_tracer(obj):
                # This is a shared arg, already has a tracer from previous
                # tracing. Delete the tracer.
                del obj.__dict__[proxy_slot]
            return obj
        else:
            return arg

    print("=== tracing ", node.name, node.target, node.args, node.kwargs)
    args = tree_map(remap_arg, node.args)
    # kwargs in this set of tests are all constants
    kwargs = node.kwargs

    # run dispatch once to get the real DTensor output
    out = operator_dispatch(
        node.target,
        args,
        node.kwargs,  # kwargs in this set of tests are all constants
        DTensor._op_to_rules,
        DTensor._custom_dispatch_ops,
    )
    node_to_obj[node] = out
    print("=== dtensor out, ", out.placements, out.size())

    # get DTensor specs for inputs and outputs
    (
        target_schema,
        redistribute,
        output_sharding,
    ) = propagate_input_sharding(
        node.target,
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
        node.target,
        kwargs=kwargs,
        specs=specs,
    )

    def unwrap_local(e):
        return e._local_tensor if isinstance(e, DTensor) else e

    return make_fx(dispatch)(tree_map(unwrap_local, args))



class SPMD(nn.Module):
    def __init__(self, module: nn.Module, schema: Schema):
        super().__init__()
        assert schema.placements == [Replicate()], (
            "SPMD only support Replicate() parameters for now"
        )
        self._local_module = module
        self._schema = schema

        # TODO: add schema_override

        self._compiled_m = None

    def _compile(self, gm: fx.GraphModule, inps: List[torch.Tensor]) -> fx.GraphModule:
        def is_param(t: torch.Tensor) -> bool:
            # N.B.: id(t) and id(param) does not match
            return t.storage().data_ptr() in [p.storage().data_ptr() for p in self._local_module.parameters()]

        # HACK: use pytree order of params to map to primals, and save the info
        # for compile_bwd.
        """
        def to_param(model: nn.Module, primal_name: str) -> torch.nn.Parameter:
            idx = int(primal_name.split("_")[-1]) - 1
            params = [p for _, p in list(tree_flatten(model.named_parameters())[0][0])]
            return params[idx] if idx < len(params) else None
        """

        node_to_obj: Dict[fx.Node, object] = {}
        # map local op node in traced_f to its corresponding subgraph of
        # DTensor ops.
        replacements: Dict[torch.fx.Node, torch.fx.GraphModule] = {}

        for i, node in enumerate(gm.graph.nodes):
            if node.op == "placeholder":
                assert i < len(inps)
                #p = to_param(self._local_module, node.name)
                if is_param(inps[i]):
                    node_to_obj[node] = DTensor.from_local(
                        inps[i], self._schema.mesh, self._schema.placements
                    )
                else:
                    node_to_obj[node] = DTensor.from_local(
                        inps[i], self._schema.mesh, [Shard(0)]
                    )
            elif isinstance(node.target, torch._ops.OpOverload):
                replacements[node] = _get_dtensor_dispatch_graph(
                    node, node_to_obj
                )
            elif node.op == "output":
                new_args = []
                for a in node.args[0]:
                    if not isinstance(a, fx.Node):
                        new_args.append(a)
                        continue
                    obj = node_to_obj[a]
                    if isinstance(obj, DTensor) and isinstance(obj.placements[0], _Partial):
                        def dummy_add(grad, zero) -> torch.Tensor:
                            return grad + zero

                        grad = obj._local_tensor
                        zero = torch.zeros_like(obj._local_tensor)

                        traced_add = make_fx(dummy_add)(grad, zero)

                        placeholders = [n for n in traced_add.graph.nodes if n.op == "placeholder"]
                        call_functions = [n for n in traced_add.graph.nodes if n.op == "call_function"]
                        assert len(placeholders) == 2
                        assert len(call_functions) == 1
                        node_to_obj[placeholders[0]] = obj
                        node_to_obj[placeholders[1]] = zero
                        traced_dispatch = _get_dtensor_dispatch_graph(call_functions[0], node_to_obj)
                        traced_dispatch.graph.lint()

                        wait = [n for n in traced_dispatch.graph.nodes if n.name == "wait_comm"]
                        add = [n for n in traced_dispatch.graph.nodes if n.name == "add"]
                        assert len(wait) == 1 and len(add) == 1
                        add[0].replace_all_uses_with(wait[0])
                        traced_dispatch.graph.lint()
                        traced_dispatch.graph.eliminate_dead_code()

                        #traced_dispatch.graph.print_tabular()


                        value_remap: Dict[fx.Node, fx.Node] = {}
                        for dtn in traced_dispatch.graph.nodes:
                            if dtn.op == "placeholder":
                                # do nothing, ignore placeholders, as it has already
                                # been prepared in value_remap
                                value_remap[dtn] = a
                            elif dtn.op == "output":
                                assert (
                                    len(dtn.args) == 1 and len(dtn.args[0]) == 1
                                ), f"Expecting single output, but got {dtn.args}"
                                #a.replace_all_uses_with(value_remap[dtn.args[0][0]])
                                new_args.append(value_remap[dtn.args[0][0]])
                            else:
                                if dtn.op == "get_attr":
                                    setattr(gm, dtn.target, getattr(traced_dispatch, dtn.target))
                                with gm.graph.inserting_before(node):
                                    value_remap[dtn] = gm.graph.node_copy(
                                        dtn, lambda n: value_remap[n]
                                    )

                gm.graph.erase_node(node)
                gm.graph.output(new_args)
                break
            else:
                raise ValueError(f"Unrecognized node {node}")

        #gm.graph.print_tabular()

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
        if dist.get_rank() == 0:
            gm.graph.print_tabular()

        return gm


    def forward(self, *args, **kwargs):
        if self._compiled_m is None:
            self._compiled_m = aot_module(
                self._local_module,
                self._compile,
                self._compile,
            )

        return self._compiled_m(*args, **kwargs)



def run(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    m = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
    device = torch.device("cpu")
    schema = Schema(
        mesh=DeviceMesh(device.type, torch.arange(2)),
        placements=[Replicate()],
    )
    spmd = SPMD(m, schema=schema)
    spmd(torch.ones(2, 10) * rank).sum().backward()
    for p in m.parameters():
        p.grad.zero_()

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    spmd(torch.ones(2, 10) * rank).sum().backward()


    print([p.grad for p in m.parameters()])


if __name__=="__main__":
    world_size = 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )