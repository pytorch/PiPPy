# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.distributed_c10d import (
    ProcessGroup,
    new_group,
    get_world_size,
    _get_global_rank,
)
from torch.testing._internal.common_utils import run_tests
from ..test_utils import DistTensorTestBase, with_comms
from spmd.tensor import DeviceMesh, Tensor, Shard, Replicate


class DeviceMeshTest(DistTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_device_mesh_basics(self):
        # construct a cuda device mesh
        mesh = DeviceMesh(self.device_type, [0, 1, 2, 3])

        # construct from a cpu local tensor with cuda device mesh
        # should automatically convert the dist tensor to cuda
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(
            dist_tensor.local_tensor().device.type, self.device_type
        )

    @with_comms
    def test_device_mesh_context_manager(self):
        with DeviceMesh(self.device_type, list(range(self.world_size))) as mesh:
            shard_spec = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = Tensor.from_local(
                local_tensor, device_mesh=mesh, placements=shard_spec
            )

        with DeviceMesh(self.device_type, list(range(self.world_size))):
            shard_spec = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = Tensor.from_local(
                local_tensor, placements=shard_spec
            )
            replica_spec = [Replicate()]
            replica_tensor = sharded_tensor.redistribute(
                placements=replica_spec
            )
            self.assertEqual(
                replica_tensor.size(), torch.Size([3 * self.world_size, 3])
            )

    @with_comms
    def test_device_mesh_2d(self):
        mesh_tensor = torch.arange(4).reshape(2, 2)
        # construct a cuda device mesh
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()

        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < 2)
            dim_ranks = expected_ranks_by_dim[dim]

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                _get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)

        # construct a dist tensor on 2d device mesh and test if works
        shard_spec = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(
            dist_tensor.local_tensor().device.type, self.device_type
        )

        # if shard on the same tensor dimension
        # we should correctly construct the global tensor size
        shard_same_dim_spec = [Shard(0), Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_same_dim_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([12, 3]))

    @with_comms
    def test_device_mesh_2d_from_dim_groups(self):
        # construct a two dimension subgroups
        dim_groups = []
        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim_group_ranks in expected_ranks_by_dim:
            for subgroup_ranks in dim_group_ranks:
                subgroup = new_group(ranks=subgroup_ranks)
                if self.rank in subgroup_ranks:
                    dim_groups.append(subgroup)

        # construct a device mesh from the subgroups
        mesh = DeviceMesh(
            self.device_type, [[0, 1], [2, 3]], dim_groups=dim_groups
        )

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < 2)
            dim_ranks = expected_ranks_by_dim[dim]

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                _get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)

        # construct a dist tensor on 2d device mesh and test if works
        shard_spec = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(
            dist_tensor.local_tensor().device.type, self.device_type
        )

    @with_comms
    def test_device_mesh_nd(self):
        # construct a cuda device mesh
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()

        for dim, dim_group in enumerate(dim_to_subgroups):
            self.assertTrue(dim < mesh_tensor.ndim)
            dim_ranks = mesh_tensor.swapdims(-1, dim).reshape(-1, 2)
            # print(dim_ranks)
            # dim_ranks = expected_ranks_by_dim[dim]

            dim_group_size = get_world_size(dim_group)
            self.assertIsInstance(dim_group, ProcessGroup)
            self.assertEqual(dim_group_size, 2)
            global_ranks = [
                _get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            for ranks in dim_ranks:
                if self.rank in ranks:
                    self.assertEqual(global_ranks, ranks.tolist())

        # construct a dist tensor on 3d device mesh and test if works
        shard_spec = [Shard(0), Shard(1), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(
            dist_tensor.local_tensor().device.type, self.device_type
        )

        # construct a dist tensor on 3d device mesh with some shards on same dim
        shard_spec = [Shard(0), Shard(0), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([12, 3, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(
            dist_tensor.local_tensor().device.type, self.device_type
        )


if __name__ == "__main__":
    run_tests()
