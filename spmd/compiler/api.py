import logging
from typing import Dict, Optional, Sequence, Tuple

import torch.distributed as dist
import torch.nn as nn
from spmd.tensor import Placement, Replicate

from .distribute import distribute, Schema
from .distributed_graph import DistributedGraph
from .graph_optimization import DistGraphOptimization, GraphOptimization
from .log_utils import rank0_info


logger: logging.Logger = logging.getLogger(__name__)


class SPMD(nn.Module):
    # TODO: add schema_override
    def __init__(
        self,
        module: nn.Module,
        schema: Schema,
        input_schemas: Sequence[Placement] = tuple(),
        optimize_first_iter: bool = False,
        optimizations: Sequence[GraphOptimization] = tuple(),
        map_param_and_grad: bool = True,
        print_graph: bool = False,
    ) -> None:
        super().__init__()
        assert schema.placements == [
            Replicate()
        ], "SPMD only support Replicate() parameters for now"

        # TODO: coalesce broadcasts
        for p in module.parameters():
            dist.broadcast(p, src=0)

        self._param_schema = schema
        self._input_schemas = input_schemas
        self._orig_module = module
        self._compiled_m: Optional[nn.Module] = None
        self._dist_graph = DistributedGraph(orig_module=module)
        self._graph_optimization = DistGraphOptimization(self._dist_graph)
        self._optimize_first_iter = optimize_first_iter
        self._optimizations = optimizations
        self._map_param_and_grad = False
        self._print_graph = print_graph

    def forward(
        self, *args: Tuple[object], **kwargs: Dict[str, object]
    ) -> object:
        if self._compiled_m is None:
            self._compiled_m = distribute(
                self._dist_graph,
                self._param_schema,
                self._input_schemas,
                self._optimize_first_iter,
                self._map_param_and_grad,
                *args,
                **kwargs,
            )

        if self._dist_graph.bwd_graph_modules:
            if not self._graph_optimization.optimized and self._optimizations:
                # Profile the module. Right now it will use the saved orig_module
                # to profile. There will be another compilation for the profiling
                # purpose.
                self._dist_graph.profile(*args, **kwargs)
                self._dist_graph.update()

                # Apply the graph optimizations if the graph is not optimized both
                # fwd and bwd graphs are ready. All optimizations should be directly
                # applied to the saved fwd and bwd gm.
                self._graph_optimization.apply(
                    self._optimizations, print_graph=self._print_graph
                )
            elif self._print_graph:
                # Graph optimization will print out the graphs. But if users do not
                # apply any optimization, we still need to print out the graph.
                fwd_gm = self._dist_graph.fwd_graph_modules[0]
                bwd_gm = self._dist_graph.bwd_graph_modules[0]
                rank0_info(logger, "The forward graph.")
                rank0_info(
                    logger, f"\n {fwd_gm.print_readable(print_output=False)}"
                )
                rank0_info(logger, "The backward graph.")
                rank0_info(
                    logger, f"\n {bwd_gm.print_readable(print_output=False)}"
                )

            # We only print out the graph once.
            self._print_graph = False

        assert self._compiled_m is not None
        return self._compiled_m(*args, **kwargs)
