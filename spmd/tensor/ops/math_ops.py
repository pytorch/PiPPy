# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed.distributed_c10d import ReduceOp
from spmd.tensor.api import Tensor
from spmd.tensor.placement_types import (
    Replicate,
    _Partial,
    is_partial,
    is_replicate,
    is_shard,
)
from spmd.tensor.ops.utils import register_impl


@register_impl("aten.sum.default")
def dist_sum(self: Tensor) -> Tensor:
    self_local = self.local_tensor()
    self_placement = self.placements[0]
    device_mesh = self.device_mesh

    local_sum = self_local.sum()

    if is_shard(self_placement) or is_partial(self_placement):
        placements = [_Partial(ReduceOp.SUM)]
        # partial reduce
        partial_sum = Tensor.from_local(local_sum, device_mesh, placements)
        # all_reduce across device
        replicate_placements = [Replicate()]
        return partial_sum.redistribute(device_mesh, replicate_placements)
    elif is_replicate(self_placement):
        return Tensor.from_local(
            local_sum, device_mesh=device_mesh, placements=self.placements
        )
    else:
        raise RuntimeError("Not supported!")
