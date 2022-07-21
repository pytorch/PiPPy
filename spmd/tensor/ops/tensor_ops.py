# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from spmd.tensor.api import Tensor
from spmd.tensor.ops.utils import register_impl


@register_impl("aten.detach.default")
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def dist_detach(self):
    device_mesh = self.device_mesh

    detached_tensor = self.local_tensor().detach()
    return Tensor.from_local(detached_tensor, device_mesh, self.placements)


@register_impl("aten.ones_like.default")
# pyre-fixme[3]: Return type must be annotated.
def dist_ones_like(
    # pyre-fixme[2]: Parameter must be annotated.
    self,
    # pyre-fixme[2]: Parameter must be annotated.
    dtype=None,
    # pyre-fixme[2]: Parameter must be annotated.
    layout=None,
    # pyre-fixme[2]: Parameter must be annotated.
    device=None,
    # pyre-fixme[2]: Parameter must be annotated.
    pin_memory=None,
    # pyre-fixme[2]: Parameter must be annotated.
    memory_format=None,
):
    device_mesh = self.device_mesh

    new_local_tensor = torch.ones_like(self.local_tensor())
    return Tensor.from_local(new_local_tensor, device_mesh, self.placements)


# @register_impl("aten.expand.default")
# def dist_expand(types, args=(), kwargs=None):
#     self_tensor = args[0]
#     device_mesh = self_tensor.device_mesh

#     new_local_tensor = torch.ones_like(self_tensor.local_tensor())
#     return Tensor.from_local(new_local_tensor, device_mesh, self_tensor.placements)
