from typing import cast, Dict, Optional, Tuple

import torch.distributed as dist
import torch.nn as nn
from spmd.tensor.placement_types import Replicate

from .bucketing_strategies import BucketingStrategy
from .distribute import distribute, Schema
from .distributed_graph import DistributedGraph
from .graph_optimization import DistGraphOptimization
from .scheduling_policies import SchedulingPolicy


class SPMD(nn.Module):
    # TODO: add schema_override
    def __init__(
        self, module: nn.Module, schema: Schema, fw_only: bool = False
    ) -> None:
        super().__init__()
        assert schema.placements == [
            Replicate()
        ], "SPMD only support Replicate() parameters for now"

        # TODO: coalesce broadcasts
        for p in module.parameters():
            dist.broadcast(p, src=0)

        self._param_schema: Schema = schema
        self._orig_module = module
        self._compiled_m: Optional[nn.Module] = None
        self._dist_graph = DistributedGraph(orig_module=module)
        self._fw_only = fw_only

    def forward(
        self, *args: Tuple[object], **kwargs: Dict[str, object]
    ) -> object:
        if self._compiled_m is None:
            # Trace and distribute the graphs.
            # A profiling to the origin_module may be required if we would like
            # to automatically decide how to distribute the module.
            self._compiled_m, fwd_gm, bwd_gm = distribute(
                self._orig_module,
                self._param_schema,
                self._fw_only,
                *args,
                **kwargs
            )
            self._dist_graph.fwd_graph_modules.append(fwd_gm)
            self._dist_graph.bwd_graph_modules.append(bwd_gm)
            # Profile the module. Right now it will use the saved orig_module to
            # profile. There will be another compilation for the profiling purpose.
            self._dist_graph = self._dist_graph.update().profile(
                *args, **kwargs
            )
            # Apply the graph optimizations. All optimizations should be directly
            # applied to the saved fwd and bwd gm.
            DistGraphOptimization(self._dist_graph).fuse_communication(
                BucketingStrategy.FIXED, SchedulingPolicy.FCFS
            )

        return cast(nn.Module, self._compiled_m)(*args, **kwargs)
