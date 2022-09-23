# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.distributed_c10d import (
    ProcessGroup,
    new_group,
    get_global_rank,
    get_world_size,
)
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd.tensor import DeviceMesh, DTensor, Shard, Replicate


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
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_device_mesh_context_manager(self):
        with DeviceMesh(self.device_type, list(range(self.world_size))) as mesh:
            shard_spec = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(
                local_tensor, device_mesh=mesh, placements=shard_spec
            )

        with DeviceMesh(self.device_type, list(range(self.world_size))):
            shard_spec = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(
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
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)

        # construct a dist tensor on 2d device mesh and test if works
        shard_spec = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

        # if shard on the same tensor dimension
        # we should correctly construct the global tensor size
        shard_same_dim_spec = [Shard(0), Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(
            local_tensor, mesh, shard_same_dim_spec
        )
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
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            current_rank_expected_group_ranks = (
                dim_ranks[0] if self.rank in dim_ranks[0] else dim_ranks[1]
            )
            self.assertEqual(global_ranks, current_rank_expected_group_ranks)

        # construct a dist tensor on 2d device mesh and test if works
        shard_spec = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_device_mesh_dim_groups_error(self):
        # construct a two dimension subgroups
        dim_groups = []
        expected_ranks_by_dim = [[[0, 2], [1, 3]], [[0, 1], [2, 3]]]
        for dim_group_ranks in expected_ranks_by_dim:
            for subgroup_ranks in dim_group_ranks:
                subgroup = new_group(ranks=subgroup_ranks)
                if self.rank in subgroup_ranks:
                    dim_groups.append(subgroup)

        if len(dim_groups) > 0:
            # dim_groups is not a list
            self.assertRaises(
                RuntimeError,
                DeviceMesh,
                self.device_type,
                [[0, 1], [2, 3]],
                dim_groups=dim_groups[0],
            )

            # dim_groups is a list, but not a list of ProcessGroup
            self.assertRaises(
                RuntimeError,
                DeviceMesh,
                self.device_type,
                [[0, 1], [2, 3]],
                dim_groups=[dim_groups[0], "dummy"],
            )

            # dim_groups has incorrect length
            self.assertRaises(
                RuntimeError,
                DeviceMesh,
                self.device_type,
                [[0, 1], [2, 3]],
                dim_groups=[dim_groups[0]],
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
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            for ranks in dim_ranks:
                if self.rank in ranks:
                    self.assertEqual(global_ranks, ranks.tolist())

        # construct a dist tensor on 3d device mesh and test if works
        shard_spec = [Shard(0), Shard(1), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

        # construct a dist tensor on 3d device mesh with some shards on same dim
        shard_spec = [Shard(0), Shard(0), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([12, 3, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)


class DeviceMeshCollectiveTest(DistTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_all_reduce_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        reduced_tensor = mesh.all_reduce(local_tensor, mesh_dim=0)
        res_num = ((0 + self.world_size - 1) * self.world_size) / 2
        self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_broadcast_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        broadcasted_tensor = mesh.broadcast(local_tensor, mesh_dim=0)
        self.assertEqual(broadcasted_tensor, torch.zeros(3, 3))

    @with_comms
    def test_scatter_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        tensor_list = [
            torch.ones(3, 3, device=self.device_type) * rank
            for rank in range(self.world_size)
        ]
        scattered_tensor = mesh.scatter(tensor_list, mesh_dim=0)
        self.assertEqual(scattered_tensor, torch.ones(3, 3) * self.rank)

        # slight complicate use case
        # fix the random seed
        torch.manual_seed(0)
        global_tensor = torch.randn(
            3, 3 * self.world_size, device=self.device_type
        )
        # chunk by dim 1 will make it non-contiguous
        chunked_list = global_tensor.chunk(self.world_size, dim=1)
        # test if scatter still works
        local_tensor = mesh.scatter(chunked_list, mesh_dim=0)
        self.assertEqual(local_tensor.shape, torch.Size((3, 3)))
        self.assertEqual(local_tensor, chunked_list[self.rank])

    @with_comms
    def test_all_gather_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dims_to_gather = [0, 1]
        for dim in dims_to_gather:
            output_size = [3, 3]
            output_size[dim] *= self.world_size
            # each rank have its own tensor, all_gather gives a list
            local_tensor = torch.ones(3, 3, device=self.device_type)
            gathered_tensor = mesh.all_gather(
                local_tensor, output_size, mesh_dim=0, tensor_dim=dim
            )
            self.assertEqual(gathered_tensor, torch.ones(output_size))

    @with_comms
    def test_reduce_scatter_1d(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dims_to_scatter = [0, 1]
        for dim in dims_to_scatter:
            input_size = [3, 3]
            input_size[dim] *= self.world_size

            input_tensor = (
                torch.ones(input_size, device=self.device_type) * self.rank
            )
            res_num = ((0 + self.world_size - 1) * self.world_size) / 2
            reduce_scattered_tensor = mesh.reduce_scatter(
                input_tensor, mesh_dim=0, tensor_dim=dim
            )
            self.assertEqual(
                reduce_scattered_tensor, torch.ones(3, 3) * res_num
            )

    @with_comms
    def test_all_gather_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            gathered_tensor = mesh.all_gather(
                local_tensor, (dim_group_size * 3, 3), mesh_dim=dim
            )
            exp_tensor = torch.ones(3 * dim_group_size, 3)
            for i in range(len(global_ranks)):
                exp_tensor[i * 3 : (i + 1) * 3] = (
                    torch.ones(3, 3) * global_ranks[i]
                )
            self.assertEqual(gathered_tensor, exp_tensor)

    @with_comms
    def test_reduce_scatter_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            local_tensor = (
                torch.ones(dim_group_size * 3, 3, device=self.device_type)
                * self.rank
            )
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            reduce_scattered_tensor = mesh.reduce_scatter(
                local_tensor, mesh_dim=dim
            )
            res_num = torch.sum(torch.tensor(global_ranks))
            self.assertEqual(
                reduce_scattered_tensor, torch.ones(3, 3) * res_num
            )

    @with_comms
    def test_all_reduce_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            reduced_tensor = mesh.all_reduce(local_tensor, mesh_dim=dim)
            res_num = sum(global_ranks)
            self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_broadcast_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            received_tensor = mesh.broadcast(local_tensor, mesh_dim=dim)
            res_num = global_ranks[0]
            self.assertEqual(received_tensor, torch.ones(3, 3) * res_num)

    @with_comms
    def test_scatter_nd(self):
        mesh_tensor = torch.arange(8).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]
            received_tensor = mesh.scatter(scattered_tensors, mesh_dim=dim)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)


if __name__ == "__main__":
    run_tests()
