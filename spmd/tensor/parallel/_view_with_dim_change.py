# Copyright (c) Meta Platforms, Inc. and affiliates
import math

import torch
import spmd.tensor.api as spmd_tensor
from spmd.tensor import DTensor as DT
from spmd.tensor.placement_types import Shard


def _view_with_sharding_dim_change(tensor, sharding_dim, shape):
    """
    Sometimes we want to change the implicit sharding dim for a
    distributed tensor without comms.
    """
    if isinstance(tensor, DT):
        return _ViewAndRedistribute.apply(tensor, sharding_dim, shape)
    else:
        return tensor.view(shape)


class _ViewAndRedistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        self: "spmd_tensor.DTensor",
        sharding_dim,
        shape,
    ):
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
    def backward(ctx, grad_output: "spmd_tensor.DTensor"):  # type: ignore
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
