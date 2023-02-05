from typing import Dict, List, Optional, Sequence, Tuple
import torch

import torch.distributed as dist
import torch.nn as nn

from spmd.tensor import Placement, Replicate

from .distribute import Schema, distribute, distribute_with_gm
from .distributed_graph import DistributedGraph
from .graph_optimization import (
    DistGraphOptimization,
    GraphOptimization,
    GraphOptimizationType,
)
import torch._dynamo as torchdynamo
from torch import fx


def spmd_dynamo_compile(fn, model, inputs, schema: Schema,
        input_schemas: Sequence[Placement] = tuple(),):
    def dynamo_compiler(gm: fx.GraphModule, 
        *args: Tuple[object],
        **kwargs: Dict[str, object],):
            
            if self._compiled_m is None:
                self._compiled_m = distribute_with_gm(
                    gm,
                    schema,
                    input_schemas,
                    *args,
                    **kwargs,
                )
            return self._compiled_m
            
            return gm

    optimize_ctx = torchdynamo.optimize(dynamo_compiler)

    # TODO: Add BW pass

    return optimize_ctx(fn)(model, inputs)


class SPMD(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        schema: Schema,
        input_schemas: Sequence[Placement] = tuple(),
        optimize_first_iter: bool = False,
        optimizations: Sequence[GraphOptimization] = tuple(),
        example_inputs=None,
    ) -> None:
        """
        Given a non-distributed nn.Module, distribute the module and apply
        optimizations over the distributed module (fx.GraphModule).

        Args:
            module (nn.Module): The target module.
            schema (Schema): The distributed schema.
            input_schemas (Sequence[Placement]): The schemas of the inputs.
            optimize_first_iter (bool): If true, SPMD will call the forward
               and backward passes to eagerly get the graphs. This can be
               problematic since SPMD currently assumes a simple out of
               the tensor and performs ``sum()`` to get the loss.
            optimizations (Sequence[GraphOptimization]): List of optimization passes
               to perform on the distributed graph module post transformation.
        """
        super().__init__()
        assert schema.placements == [
            Replicate()
        ], "SPMD only support Replicate() parameters for now"

        # TODO: Fix model initialization with coalescing.
        # This needs to happen post model transformation.
        # Consider an explicit model init API.
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
                **kwargs,
            )

        if self._dist_graph.bwd_graph_modules:
            if not self._graph_optimization.optimized and self._optimizations:
                # Profile the module. Right now it will use the saved orig_module
                # to profile. There will be another compilation for the profiling
                # purpose.

                # Gate the profiling call until it is fully verified with different
                # models.
                if (
                    GraphOptimization(GraphOptimizationType.NOOP)
                    in self._optimizations
                ):
                    self._dist_graph.profile(*args, **kwargs)
                self._dist_graph.update()

                # Apply the graph optimizations. If the graph is not optimized, both
                # fwd and bwd graphs are ready. All optimizations should be directly
                # applied to the saved fwd and bwd gm.
                self._graph_optimization.apply(self._optimizations)

        assert self._compiled_m is not None
        return self._compiled_m(*args, **kwargs)
