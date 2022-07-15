# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed.distributed_c10d import (
    ReduceOp
)
from spmd.tensor.api import Tensor
from spmd.tensor.placement_types import Shard, Replicate, _Partial
from spmd.tensor.ops.utils import register_impl


@register_impl("aten.sum.default")
def dist_sum(self: Tensor) -> Tensor:
    self_local = self.local_tensor()
    self_placement = self.placements[0]
    device_mesh = self.device_mesh

    local_sum = self_local.sum()

    if isinstance(self_placement, Shard) or isinstance(self_placement, _Partial):
        placements = [_Partial(ReduceOp.SUM)]
        # partial reduce
        partial_sum = Tensor.from_local(local_sum, device_mesh, placements)
        # all_reduce across device
        replicate_placements = [Replicate()]
        return partial_sum.redistribute(device_mesh, replicate_placements)
    elif isinstance(self_placement, Replicate):
        return Tensor.from_local(local_sum, device_mesh=device_mesh, placements=self.placements)
    else:
        raise RuntimeError("Not supported!")
