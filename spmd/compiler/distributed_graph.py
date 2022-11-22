from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import fx


class Profiler:
    """
    This is a placeholder class to allow the definition of DistributedGraph.
    Will pull in the profiler implementation later.
    """

    pass


class DistributedGraph:
    def __init__(
        self,
        orig_module: Optional[nn.Module] = None,
    ) -> None:
        self.orig_module: Optional[nn.Module] = orig_module
        self.fwd_graph_modules: List[fx.GraphModule] = []
        self.fwd_profilers: List[Profiler] = []
        self.bwd_graph_modules: List[fx.GraphModule] = []
        self.bwd_profilers: List[Profiler] = []
        self.fwd_aot_module_inputs: List[List[torch.Tensor]] = []
        self.bwd_aot_module_inputs: List[List[torch.Tensor]] = []
        # HACK: ideally, it will be better if fx/AOTAutograd can provide a way
        # to access original param, instead of us keeping the following maps.
        self.primal_name_to_node: List[Dict[str, fx.Node]] = []
        self.primal_to_param: List[Dict[fx.Node, nn.Parameter]] = []
        self.grad_to_primal: List[Dict[fx.Node, fx.Node]] = []

        self._dirty = True

    def _map_primal_grad(self) -> None:
        if len(self.primal_to_param) == len(self.fwd_graph_modules):
            # Already connect the mapping.
            return

        def to_param(model: nn.Module, primal_name: str) -> nn.Parameter:
            idx = int(primal_name.split("_")[-1]) - 1
            # HACK: Dynamo primal order is the reverse of AOTAutograd???
            params = [
                p
                for _, p in reversed(
                    list(pytree.tree_flatten(model.named_parameters())[0][0])
                )
            ]
            return params[idx] if idx < len(params) else None

        for (fwd_gm, bwd_gm) in zip(
            self.fwd_graph_modules,
            self.bwd_graph_modules,
        ):
            self.primal_to_param.append([])
            self.primal_name_to_node.append([])
            self.grad_to_primal.append([])
            primal_to_param = self.primal_to_param[-1]
            primal_name_to_node = self.primal_name_to_node[-1]
            grad_to_primal = self.grad_to_primal[-1]
            for node in bwd_gm.graph.nodes:
                if node.op == "placeholder" and node.target.startswith(
                    "primal"
                ):
                    param = to_param(fwd_gm, node.name)
                    if param is not None:
                        assert (
                            node not in self.primal_to_param
                        ), f"inserting {node.target} twice"
                        # HACK: use sub-graph gid to distinguish primals with
                        # the same name
                        primal_to_param[node] = param
                        primal_name_to_node[node.target] = node

            # HACK: today, there is no good way to map AOTAutograd primals back
            # to parameters in the original model. The current implementation
            # relies on the implicit AOTAutograd behavior that primals match the
            # order of params in pytree(model.named_parameters()), and grad
            # output in the backward graph matches the same order. So we count
            # the number of params and use that to access primals and grads in
            # the fwd/bwd graphs.
            n_grads = sum([p.requires_grad for p in fwd_gm.parameters()])
            for node in bwd_gm.graph.nodes:
                if node.op != "output":
                    continue

                for i, grad_node in enumerate(node.args[0][:n_grads]):
                    primal = f"primals_{i+1}"
                    primal_node = primal_name_to_node[primal]
                    grad_to_primal[grad_node] = primal_node
                break

    def update(self) -> "DistributedGraph":
        self._map_primal_grad()
        self._dirty = False
        return self

    def profile(self) -> "DistributedGraph":
        # TODO: profile model
        return self

    def validate(self) -> None:
        assert (
            not self._dirty
        ), "The graph is modified but ``update()`` is not called to update the information."
        assert (
            len(self.fwd_graph_modules) == 1
        ), "DistributedGraph has not support multiple subgraphs yet."
        assert len(self.fwd_graph_modules) == len(self.bwd_graph_modules)

        assert (
            len(self.primal_name_to_node)
            == len(self.primal_to_param)
            == len(self.grad_to_primal)
        )
        assert len(self.primal_name_to_node) <= len(self.fwd_graph_modules)
        assert len(self.fwd_profilers) == len(self.bwd_profilers)
        assert len(self.fwd_profilers) <= len(self.fwd_graph_modules)
