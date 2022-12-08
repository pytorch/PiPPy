from typing import Dict, Optional, Sequence, Tuple

import torch.distributed as dist
import torch.nn as nn
from spmd.tensor import Placement, Replicate

from .distribute import distribute, Schema
from .distributed_graph import DistributedGraph
from .graph_optimization import DistGraphOptimization, GraphOptimization
from .log_utils import get_mem_usage


class SPMD(nn.Module):
    # TODO: add schema_override
    def __init__(
        self,
        module: nn.Module,
        schema: Schema,
        input_schemas: Sequence[Placement] = tuple(),
        optimize_first_iter: bool = False,
        optimizations: Sequence[GraphOptimization] = tuple(),
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

    def forward(
        self, *args: Tuple[object], **kwargs: Dict[str, object]
    ) -> object:
        if self._compiled_m is None:
            self._compiled_m = distribute(
                self._dist_graph,
                self._param_schema,
                self._input_schemas,
                self._optimize_first_iter,
                *args,
                **kwargs
            )

        if (
            not self._graph_optimization.optimized
            and self._dist_graph.bwd_graph_modules
            and self._optimizations
        ):
            # Profile the module. Right now it will use the saved orig_module
            # to profile. There will be another compilation for the profiling
            # purpose.
            self._dist_graph.profile(*args, **kwargs)
            self._dist_graph.update()

            # Apply the graph optimizations if the graph is not optimized both
            # fwd and bwd graphs are ready. All optimizations should be directly
            # applied to the saved fwd and bwd gm.
            self._graph_optimization.apply(self._optimizations)

        assert self._compiled_m is not None
        print(f"pre FW {get_mem_usage()}")
        return self._compiled_m(*args, **kwargs)
