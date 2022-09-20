# Copyright (c) Meta Platforms, Inc. and affiliates
import math

import torch
import spmd.tensor.api as spmd_tensor
from spmd.tensor import DTensor as DT
from spmd.tensor.placement_types import Shard
from typing import Tuple, Union


def _view_with_sharding_dim_change(
    tensor: Union[torch.Tensor, DT], sharding_dim: int, shape: Tuple[int, ...]
) -> Union[torch.Tensor, DT]:
    """
    Sometimes we want to change the implicit sharding dim for a
    distributed tensor without comms.
    """
    if isinstance(tensor, DT):
        # pyre-fixme[16]: Undefined attribute.
        return _ViewAndRedistribute.apply(tensor, sharding_dim, shape)
    else:
        return tensor.view(shape)


class _ViewAndRedistribute(torch.autograd.Function):
    @staticmethod
    # pyre-fixme[14]: Inconsistent override.
    def forward(
        ctx,  # type: ignore
        self: DT,
        sharding_dim: int,
        shape: Tuple[int, ...],
    ) -> DT:
        ctx.previous_placement = self.placements
        ctx.previous_device_mesh = self.device_mesh
        ctx.previous_local_shape = self.to_local().size()
        assert (
            self.device_mesh.ndim == 1
        ), "Only support 1D Device Mesh for _ViewAndRedistribute."
        if (
            self.placements[0].is_shard(dim=sharding_dim)
            or self.placements[0].is_replicate()
            or self.placements[0].is_partial()
        ):
            # pyre-fixme[7]: Incompatible return type.
            return self.view(shape)
        else:
            if sharding_dim < 0:
                sharding_dim += self.dim()

            device_mesh = self.device_mesh
            world_size = device_mesh.size(dim=0)
            new_sharding_placement = [Shard(sharding_dim)]

            # Fix shape
            try:
                infer_idx = shape.index(-1)
            except ValueError:
                infer_idx = None

            # Infer the dim which is specified with -1.
            if infer_idx is not None:
                st_size = math.prod(self.size())  # type: ignore[attr-defined]
                shape_size = -1 * math.prod(shape)  # type: ignore[attr-defined]
                # pyre-fixme[60]: Concatenation not yet support for multiple variadic
                shape = (
                    *shape[:infer_idx],
                    st_size // shape_size,
                    *shape[infer_idx + 1 :],
                )

            # pyre-fixme[60]: Concatenation not yet support for multiple variadic
            new_local_tensor_size = (
                *shape[:sharding_dim],
                shape[sharding_dim] // world_size,
                *shape[sharding_dim + 1 :],
            )
            new_local_tensor = self.to_local().view(*new_local_tensor_size)

            return spmd_tensor.DTensor(
                new_local_tensor,
                device_mesh,
                new_sharding_placement,
                requires_grad=new_local_tensor.requires_grad,
            )

    @staticmethod
    def backward(ctx, grad_output: DT) -> Tuple[DT, None, None]:  # type: ignore
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh
        previous_local_tensor_size = ctx.previous_local_shape
        return (
            spmd_tensor.DTensor(
                grad_output.to_local().view(*previous_local_tensor_size),
                previous_device_mesh,
                previous_placement,
                requires_grad=grad_output.requires_grad,
            ),
            None,
            None,
        )
