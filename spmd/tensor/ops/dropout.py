# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from typing import Tuple
from spmd.tensor.ops.utils import register_impl
from spmd.tensor.api import Tensor


@register_impl("aten.native_dropout.default")
def _dist_dropout(self: Tensor, p: float, train: bool) -> Tuple[Tensor, Tensor]:
    self_placement = self.placements[0]
    # TODO: To figure out why partial tensor does not dispatch here when in CPU.
    # and with kwargs.
    if self_placement.is_partial() or self_placement.replicate():
        raise RuntimeError("Not supported!")
    else:
        local_tensor, mask = torch.ops.aten.native_dropout(
            self.local_tensor(), p=p, train=train
        )
        return (
            Tensor.from_local(
                local_tensor,
                device_mesh=self.device_mesh,
                placements=self.placements,
            ),
            Tensor.from_local(
                mask, device_mesh=self.device_mesh, placements=self.placements
            ),
        )
