# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.distributed_c10d import (
    ProcessGroup,
    GroupMember,
    new_group,
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
        self.assertEqual(dist_tensor.local_tensor().device.type, self.device_type)

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
            sharded_tensor = Tensor.from_local(local_tensor, placements=shard_spec)
            replica_spec = [Replicate()]
            replica_tensor = sharded_tensor.redistribute(placements=replica_spec)
            self.assertEqual(
                replica_tensor.size(), torch.Size([3 * self.world_size, 3])
            )

    @with_comms
    def test_device_mesh_2d(self):
        # construct a cuda device mesh
        mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])

        # check first dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        first_dim_groups = dim_to_subgroups[0]
        self.assertEqual(len(first_dim_groups), 2)
        # first dim subgroups should be [0, 2], [1, 3]
        first_dim_expected_ranks = [[0, 2], [1, 3]]
        for i, first_dim_group in enumerate(first_dim_groups):
            group_ranks = first_dim_expected_ranks[i]
            # groups will only be visable to the ranks involved
            if self.rank in group_ranks:
                self.assertIsInstance(first_dim_group, ProcessGroup)
            else:
                self.assertEqual(first_dim_group, GroupMember.NON_GROUP_MEMBER)

        # check second dim groups
        second_dim_groups = dim_to_subgroups[1]
        self.assertEqual(len(second_dim_groups), 2)
        # second dim subgroups should be [0, 1], [2, 3]
        second_dim_expected_ranks = [[0, 1], [2, 3]]
        for i, second_dim_group in enumerate(second_dim_groups):
            group_ranks = second_dim_expected_ranks[i]
            # groups will only be visable to the ranks involved
            if self.rank in group_ranks:
                self.assertIsInstance(second_dim_group, ProcessGroup)
            else:
                self.assertEqual(second_dim_group, GroupMember.NON_GROUP_MEMBER)

        # construct a dist tensor on 2d device mesh and test if works
        shard_spec = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.local_tensor().device.type, self.device_type)

    @with_comms
    def test_device_mesh_2d_from_dim_groups(self):
        # construct a two dimension subgroups
        dim_groups = []
        dim_expected_ranks = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim_group_ranks in dim_expected_ranks:
            per_dim_groups = []
            for subgroup_ranks in dim_group_ranks:
                per_dim_groups.append(new_group(ranks=subgroup_ranks))
            dim_groups.append(per_dim_groups)

        # construct a device mesh from the subgroups
        mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]], dim_groups=dim_groups)

        # check first dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        first_dim_groups = dim_to_subgroups[0]
        self.assertEqual(len(first_dim_groups), 2)
        # first dim subgroups should be [0, 2], [1, 3]
        first_dim_expected_ranks = [[0, 2], [1, 3]]
        for i, first_dim_group in enumerate(first_dim_groups):
            group_ranks = first_dim_expected_ranks[i]
            # groups will only be visable to the ranks involved
            if self.rank in group_ranks:
                self.assertIsInstance(first_dim_group, ProcessGroup)
            else:
                self.assertEqual(first_dim_group, GroupMember.NON_GROUP_MEMBER)

        # check second dim groups
        second_dim_groups = dim_to_subgroups[1]
        self.assertEqual(len(second_dim_groups), 2)
        # second dim subgroups should be [0, 1], [2, 3]
        second_dim_expected_ranks = [[0, 1], [2, 3]]
        for i, second_dim_group in enumerate(second_dim_groups):
            group_ranks = second_dim_expected_ranks[i]
            # groups will only be visable to the ranks involved
            if self.rank in group_ranks:
                self.assertIsInstance(second_dim_group, ProcessGroup)
            else:
                self.assertEqual(second_dim_group, GroupMember.NON_GROUP_MEMBER)

        # construct a dist tensor on 2d device mesh and test if works
        shard_spec = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.local_tensor().device.type, self.device_type)

    @with_comms
    def test_device_mesh_nd(self):
        # construct a cuda device mesh
        mesh = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh.tolist())

        # check first dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        first_dim_groups = dim_to_subgroups[0]
        self.assertEqual(len(first_dim_groups), 4)
        self.assertEqual(len(dim_to_subgroups[2]), 4)

        # check first dim groups
        first_dim_expected_ranks = [[0, 4], [1, 5], [2, 6], [3, 7]]
        for i, first_dim_group in enumerate(first_dim_groups):
            group_ranks = first_dim_expected_ranks[i]
            # groups will only be visable to the ranks involved
            if self.rank in group_ranks:
                self.assertIsInstance(first_dim_group, ProcessGroup)
            else:
                self.assertEqual(first_dim_group, GroupMember.NON_GROUP_MEMBER)

        # check second dim groups
        second_dim_groups = dim_to_subgroups[1]
        self.assertEqual(len(second_dim_groups), 4)
        second_dim_expected_ranks = [[0, 2], [1, 3], [4, 6], [5, 7]]
        for i, second_dim_group in enumerate(second_dim_groups):
            group_ranks = second_dim_expected_ranks[i]
            # groups will only be visable to the ranks involved
            if self.rank in group_ranks:
                self.assertIsInstance(second_dim_group, ProcessGroup)
            else:
                self.assertEqual(second_dim_group, GroupMember.NON_GROUP_MEMBER)

        # check third dim groups
        third_dim_groups = dim_to_subgroups[2]
        self.assertEqual(len(third_dim_groups), 4)
        third_dim_expected_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]
        for i, third_dim_group in enumerate(third_dim_groups):
            group_ranks = third_dim_expected_ranks[i]
            # groups will only be visable to the ranks involved
            if self.rank in group_ranks:
                self.assertIsInstance(third_dim_group, ProcessGroup)
            else:
                self.assertEqual(third_dim_group, GroupMember.NON_GROUP_MEMBER)

        # construct a dist tensor on 3d device mesh and test if works
        shard_spec = [Shard(0), Shard(1), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = Tensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.local_tensor().device.type, self.device_type)


if __name__ == "__main__":
    run_tests()
