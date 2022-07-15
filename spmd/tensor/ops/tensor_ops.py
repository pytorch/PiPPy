# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from spmd.tensor.api import Tensor
from spmd.tensor.ops.utils import register_impl


@register_impl("aten.detach.default")
def dist_detach(self):
    device_mesh = self.device_mesh

    detached_tensor = self.local_tensor().detach()
    return Tensor.from_local(detached_tensor, device_mesh, self.placements)


@register_impl("aten.ones_like.default")
def dist_ones_like(self, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    device_mesh = self.device_mesh

    new_local_tensor = torch.ones_like(self.local_tensor())
    return Tensor.from_local(new_local_tensor, device_mesh, self.placements)

# @register_impl("aten.expand.default")
# def dist_expand(types, args=(), kwargs=None):
#     self_tensor = args[0]
#     device_mesh = self_tensor.device_mesh

#     new_local_tensor = torch.ones_like(self_tensor.local_tensor())
#     return Tensor.from_local(new_local_tensor, device_mesh, self_tensor.placements)
