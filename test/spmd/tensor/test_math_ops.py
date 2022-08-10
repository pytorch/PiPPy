# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests

from spmd.tensor.ops.prop_rules import einop_prop
from spmd.tensor.placement_types import PlacementSpec
from ..test_utils import DistTensorTestBase, with_comms
from spmd import distribute_tensor, DeviceMesh, Shard, Replicate


class DistMathOpsTest(DistTensorTestBase):
    @with_comms
    def test_einop_propagation(self):
        # plain einsum, mm
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        mat1, mat2 = [-1, -1], [-1, 0]
        mat1_spec = PlacementSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = PlacementSpec.from_dim_map(mesh, mat2, [])
        output_spec = einop_prop("mk,kn->mn", (mat1_spec, mat2_spec))
        self.assertEqual(output_spec.dim_map, [-1, 0])

        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = PlacementSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = PlacementSpec.from_dim_map(mesh, mat2, [])
        output_spec = einop_prop("mk,kn->mn", (mat1_spec, mat2_spec))
        self.assertEqual(output_spec.dim_map, [0, -1])

        mat1, mat2 = [-1, 0], [0, -1]
        mat1_spec = PlacementSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = PlacementSpec.from_dim_map(mesh, mat2, [])
        output_spec = einop_prop("mk,kn->mn", (mat1_spec, mat2_spec))
        self.assertTrue(output_spec.placements[0].is_partial())

        # addition
        output_spec = einop_prop("ij,ij->ij", (mat1_spec, mat1_spec))
        # broadcast addition
        mat1 = [-1, 0, -1]
        mat1_spec = PlacementSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = PlacementSpec.from_dim_map(mesh, [-1], [])
        output_spec = einop_prop("ijk,k->ijk", (mat1_spec, mat2_spec))
        self.assertEqual(output_spec.dim_map, [-1, 0, -1])

    @with_comms
    def test_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_sum = torch.randn(12, 8)
        sumed_tensor = tensor_to_sum.sum()
        mat1 = distribute_tensor(tensor_to_sum, device_mesh, shard_spec)
        dt_sum = mat1.sum()
        self.assertEqual(dt_sum.to_local(), sumed_tensor)


if __name__ == "__main__":
    run_tests()
