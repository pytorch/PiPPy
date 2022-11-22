from enum import auto, Enum
from functools import partial
from typing import cast, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from functorch.compile import aot_module, make_boxed_func
from spmd.tensor.placement_types import Replicate

from .distribute import Schema
from .distributed_graph import DistributedGraph
from .graph_optimization import DistGraphOptimization


class TrainingPhase(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class SPMD(nn.Module):
    # TODO: add schema_override
    def __init__(self, module: nn.Module, schema: Schema) -> None:
        super().__init__()
        assert schema.placements == [
            Replicate()
        ], "SPMD only support Replicate() parameters for now"

        # TODO: coalesce broadcasts
        for p in module.parameters():
            dist.broadcast(p, src=0)

        self._param_schema: Schema = schema
        self._local_module = module
        self._compiled_m: Optional[nn.Module] = None
        self._dist_graph = DistributedGraph(orig_module=module)

    def _compile(
        self,
        training_phase: TrainingPhase,
        gm: fx.GraphModule,
        inps: List[torch.Tensor],
    ) -> fx.GraphModule:
        if training_phase == TrainingPhase.FORWARD:
            self._dist_graph.fwd_graph_modules.append(gm)
            self._dist_graph.fwd_aot_module_inputs.append(inps)
        elif training_phase == TrainingPhase.BACKWARD:
            self._dist_graph.bwd_graph_modules.append(gm)
            self._dist_graph.bwd_aot_module_inputs.append(inps)

        return make_boxed_func(gm)

    def _compile_module(self, example_input: torch.Tensor) -> None:
        self._compiled_m = aot_module(
            self._dist_graph.orig_module,
            partial(self._compile, TrainingPhase.FORWARD),
            partial(self._compile, TrainingPhase.BACKWARD),
        )

        # Force to compile the forward
        output = cast(nn.Module, self._compiled_m)(example_input)
        # Force to compile the backward
        output.sum().backward()
        # Clear the gradient
        for p in self._compiled_m.parameters():
            if p.grad is not None:
                p.grad = None

        # Apply the profiling and graph optimization passes.
        self._dist_graph = (
            DistGraphOptimization(graph=self._dist_graph.update().profile())
            .distribute(self._param_schema)
            .graph
        )

    def forward(
        self, *args: Tuple[object], **kwargs: Dict[str, object]
    ) -> object:
        if self._compiled_m is None:
            # TODO: formalize how to get the dummy input. Currently, assume the
            # first argument is the materialized input.
            self._compile_module(args[0])

        return cast(nn.Module, self._compiled_m)(*args, **kwargs)
