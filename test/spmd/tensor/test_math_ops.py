# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests

from spmd.tensor.placement_types import Replicate
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd import distribute_tensor, DeviceMesh, Shard


class DistMathOpsTest(DistTensorTestBase):
    @with_comms
    def test_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_sum = torch.randn(12, 8, 8)

        mat1 = distribute_tensor(tensor_to_sum, device_mesh, shard_spec)

        for dim in range(tensor_to_sum.ndim):
            dim_sumed_tensor = tensor_to_sum.sum(dim=dim)
            dt_dim_sumed_tensor = mat1.sum(dim=dim).redistribute(
                device_mesh, [Replicate()] * device_mesh.ndim
            )
            self.assertEqual(dt_dim_sumed_tensor.to_local(), dim_sumed_tensor)

        full_sumed_tensor = tensor_to_sum.sum()
        dt_sum = mat1.sum().redistribute(
            device_mesh, [Replicate()] * device_mesh.ndim
        )
        self.assertEqual(dt_sum.to_local(), full_sumed_tensor)


if __name__ == "__main__":
    run_tests()
