# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed.distributed_c10d import ReduceOp
from spmd.tensor.api import DTensor
from spmd.tensor.placement_types import Replicate, _Partial
from spmd.tensor.ops.utils import register_impl


@register_impl("aten.sum.default")
def dist_sum(self: DTensor) -> DTensor:
    self_local = self.to_local()
    self_placement = self.placements[0]
    device_mesh = self.device_mesh

    local_sum = self_local.sum()

    if self_placement.is_shard() or self_placement.is_partial():
        placements = [_Partial(ReduceOp.SUM)]
        # partial reduce
        partial_sum = DTensor.from_local(local_sum, device_mesh, placements)
        # all_reduce across device
        replicate_placements = [Replicate()]
        return partial_sum.redistribute(device_mesh, replicate_placements)
    elif self_placement.is_replicate():
        return DTensor.from_local(
            local_sum, device_mesh=device_mesh, placements=self.placements
        )
    else:
        raise RuntimeError("Not supported!")
