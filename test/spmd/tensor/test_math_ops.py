# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from spmd.tensor.dispatch import OpSchema

from spmd.tensor.ops.math_ops import einop_rule, reduction_rule
from spmd.tensor.placement_types import DTensorSpec, Replicate
from spmd.test.common_utils import DistTensorTestBase, with_comms  # type: ignore
from spmd import distribute_tensor, DeviceMesh, Shard


class DistMathOpsTest(DistTensorTestBase):
    @with_comms
    def test_einop_basic_propagation(self):
        # plain einsum, mm
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # propagate col-wise sharding
        mat1, mat2 = [-1, -1], [-1, 0]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = DTensorSpec.from_dim_map(mesh, mat2, [])
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema((mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # propagate row-wise sharding
        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = DTensorSpec.from_dim_map(mesh, mat2, [])
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema((mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])

        # generate partial
        mat1, mat2 = [-1, 0], [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = DTensorSpec.from_dim_map(mesh, mat2, [])
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema((mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertTrue(output_spec.placements[0].is_partial())

    @with_comms
    def test_einop_pointwise_propagation(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # addition
        mat1 = [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        output_spec = einop_rule(
            "ij,ij->ij", OpSchema((mat1_spec, mat1_spec), {})
        )
        # broadcast addition
        mat1 = [-1, 0, -1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = DTensorSpec.from_dim_map(mesh, [-1], [])
        output_sharding = einop_rule(
            "ijk,k->ijk", OpSchema((mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0, -1])

    @with_comms
    def test_einop_merge_sharding(self):
        # 2d mesh einop merge sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)
        mat1, mat2 = [0, -1], [-1, 1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = DTensorSpec.from_dim_map(mesh, mat2, [])
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema((mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, 1])

    @with_comms
    def test_einop_linearity(self):
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [1])
        mat2_spec = DTensorSpec.from_dim_map(mesh, mat2, [])
        # if not turn on linearity, partial sum should trigger error
        with self.assertRaisesRegex(RuntimeError, "Cannot do generic op"):
            einop_rule("mk,kn->mn", OpSchema((mat1_spec, mat2_spec), {}))

        # einop prop with linearity on mm, should give back suggestion
        # on converting placements to partial
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema((mat1_spec, mat2_spec), {}), linearity=True
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.schema_suggestions
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions[0].args_schema[1]
        # mat2 mesh dim 1 should become partial now!
        self.assertTrue(mat2_spec.placements[1].is_partial())

        # einop prop with linearity on point-wise, should give back suggestion
        # on converting placements to partial
        mat1, mat2 = [0, -1], [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [1])
        mat2_spec = DTensorSpec.from_dim_map(mesh, mat2, [])

        output_sharding = einop_rule(
            "ij,ij->ij", OpSchema((mat1_spec, mat2_spec), {}), linearity=True
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.schema_suggestions
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions[0].args_schema[1]
        # mat2 mesh dim 1 should become partial now!
        self.assertTrue(mat2_spec.placements[1].is_partial())

    @with_comms
    def test_einop_errors(self):
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mat1, mat2 = [0, -1], [0, 1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = DTensorSpec.from_dim_map(mesh, mat2, [])
        with self.assertRaisesRegex(RuntimeError, "across the same mesh dim!"):
            einop_rule("mk,kn->mn", OpSchema((mat1_spec, mat2_spec), {}))

        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        mat2_spec = DTensorSpec.from_dim_map(mesh, mat2, [])

        with self.assertRaisesRegex(
            AssertionError, "sharded two different ways:"
        ):
            einop_rule("ij,ij->ij", OpSchema((mat1_spec, mat2_spec), {}))

    @with_comms
    def test_reduction_rule(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # reduction on a 2d mat
        mat1 = [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(mesh, mat1, [])
        # reduction on dim 0
        output_sharding_0 = reduction_rule(OpSchema((mat1_spec, 0), {}))
        self.assertIsNotNone(output_sharding_0.output_spec)
        self.assertEqual(output_sharding_0.output_spec.dim_map, [-1])
        # pending sum on dim 0
        self.assertEqual(output_sharding_0.output_spec.sums, [0])

        # reduction on dim 1
        output_sharding_1 = reduction_rule(OpSchema((mat1_spec, 1), {}))
        self.assertIsNotNone(output_sharding_1.output_spec)
        self.assertEqual(output_sharding_1.output_spec.dim_map, [0])
        self.assertEqual(output_sharding_1.output_spec.sums, [])

        # full reduction if not specify dim
        output_sharding_all_dim = reduction_rule(OpSchema((mat1_spec,), {}))
        self.assertIsNotNone(output_sharding_all_dim.output_spec)
        self.assertEqual(output_sharding_all_dim.output_spec.dim_map, [])
        # pending sum on mesh
        self.assertEqual(output_sharding_all_dim.output_spec.sums, [0])

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
