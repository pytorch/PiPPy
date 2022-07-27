# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from ..test_utils import DistTensorTestBase, with_comms
from spmd import distribute_tensor, DeviceMesh, Shard, Replicate


class DistMathOpsTest(DistTensorTestBase):
    @with_comms
    def test_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_sum = torch.randn(12, 8)
        sumed_tensor = tensor_to_sum.sum()
        mat1 = distribute_tensor(tensor_to_sum, device_mesh, shard_spec)
        dt_sum = mat1.sum()
        self.assertEqual(dt_sum.local_tensor(), sumed_tensor)


if __name__ == "__main__":
    run_tests()
