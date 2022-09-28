# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Tuple

import torch

from spmd.tensor.api import DTensor
from spmd.tensor.ops.utils import register_impl


@register_impl("aten.native_dropout.default")
def _dist_dropout(
    self: DTensor,
    p: float,
    train: bool,
) -> Tuple[DTensor, DTensor]:
    self_placement = self.placements[0]
    # TODO: To figure out why partial tensor does not dispatch here when in CPU.
    # and with kwargs.
    if self_placement.is_partial() or self_placement.is_replicate():
        raise RuntimeError("Not supported!")
    else:
        local_tensor, mask = torch.ops.aten.native_dropout(
            self._local_tensor, p=p, train=train
        )
        return (
            DTensor(
                local_tensor,
                device_mesh=self.device_mesh,
                placements=self.placements,
            ),
            DTensor(
                mask, device_mesh=self.device_mesh, placements=self.placements
            ),
        )
