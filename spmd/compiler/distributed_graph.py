from typing import Any, Dict, List, Optional

import torch

import torch.nn as nn
from torch import fx

from .profiler import GraphProfiler, GraphType, ProfilerEngine


class DistributedGraph:
    def __init__(
        self,
        orig_module: Optional[nn.Module] = None,
    ) -> None:
        self.orig_module: Optional[nn.Module] = orig_module
        self.fwd_graph_modules: List[fx.GraphModule] = []
        self.fwd_profilers: List[GraphProfiler] = []
        self.bwd_graph_modules: List[fx.GraphModule] = []
        self.bwd_profilers: List[GraphProfiler] = []

        # The mapping information is required for graph optimization.
        # HACK: ideally, it will be better if fx/AOTAutograd can provide a way
        # to access original param, instead of us keeping the following maps.
        self.primal_name_to_node: List[Dict[str, fx.Node]] = []
        self.primal_to_param: List[Dict[fx.Node, nn.Parameter]] = []
        self.grad_to_primal: List[Dict[fx.Node, fx.Node]] = []

        # Indicate `update()` must be called before applying any optimization.
        self._dirty = True

    def _map_param_grad(self) -> None:
        if len(self.primal_to_param) == len(self.fwd_graph_modules):
            # Already connect the mapping.
            return
        # TODO: add primal->param and grad->primal mapping. If we cannot
        # derive the mapping using the stored information (graphs and inputs),
        # then this has to be done in the tracing flow.

    def update(self) -> "DistributedGraph":
        self._map_param_grad()
        self._dirty = False
        return self

    def profile(self, *args: Any, **kwargs: Any) -> "DistributedGraph":
        """
        Profile the given distributed graph. The arguments are the inputs
        of the module as a real run is required to do profiling.
        """
        # TODO(chienchin): fix how to get the correct forward_loss
        def forward_loss(
            module: nn.Module, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            return module(*args, **kwargs).sum()

        assert self.orig_module is not None
        engine = ProfilerEngine(
            self.orig_module,
            forward_loss,
            dist_fwd_gm=self.fwd_graph_modules[0],
            dist_bwd_gm=self.bwd_graph_modules[0],
        )
        engine.run(*args, **kwargs)
        engine.summarize(to_print=True)
        self.fwd_profilers.append(engine.profilers[0][GraphType.FORWARD])
        self.bwd_profilers.append(engine.profilers[0][GraphType.BACKWARD])

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
