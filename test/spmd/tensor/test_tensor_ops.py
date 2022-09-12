# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
    TEST_GPU_NUM,
)
from spmd import distribute_tensor, DeviceMesh, DTensor, Shard, Replicate
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
import itertools


class DistTensorOpsTest(DistTensorTestBase):
    @with_comms
    def test_detach(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        detached_mat = mat.detach()
        self.assertFalse(detached_mat is mat)

    @with_comms
    def test_clone(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        specs = [[Replicate()], [Shard(0)]]
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        for spec in specs:
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            cloned_mat = mat.clone()
            self.assertFalse(cloned_mat is mat)
            self.assertEqual(cloned_mat.to_local(), mat.to_local())

    @with_comms
    def test_contiguous(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(3, 5, 6, requires_grad=True)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        self.assertTrue(dist_tensor.is_contiguous())
        # shard on dim 0 should not change stride (30, 6, 1)
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        new_dt = dist_tensor.transpose(0, 2)
        self.assertFalse(new_dt.is_contiguous())
        self.assertFalse(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(new_dt.stride(), (1, 6, 30))

        new_dt = new_dt.contiguous()
        self.assertTrue(new_dt.is_contiguous())
        self.assertTrue(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        # check backward
        new_dt.to_local().sum().backward()
        self.assertEqual(tensor.grad, torch.ones(3, 5, 6))

    @with_comms
    def test_inplace_op(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        dt_to_add = distribute_tensor(input_tensor, mesh, [Shard(0)])
        dt_to_mul = dt_to_add.clone()
        expected_add_dt = dt_to_add.clone() + 3
        add_res = dt_to_add.add_(3)
        expected_mul_dt = dt_to_mul.clone() * 3
        mul_res = dt_to_mul.mul_(3)
        # inplace op should be the same instance before and after
        self.assertTrue(add_res is dt_to_add)
        self.assertEqual(add_res.to_local(), expected_add_dt.to_local())

        self.assertTrue(mul_res is dt_to_mul)
        self.assertEqual(mul_res.to_local(), expected_mul_dt.to_local())

    @with_comms
    def test_op_out_variant(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        dist_tensor_out = distribute_tensor(input_tensor, mesh, [Shard(0)])
        expected_dt = dist_tensor_out.clone() + 3
        res = torch.add(dist_tensor_out, 3, out=dist_tensor_out)
        # op out variant should be the same instance before and after
        self.assertTrue(res is dist_tensor_out)
        self.assertEqual(dist_tensor_out.to_local(), expected_dt.to_local())

    @with_comms
    def test_empty_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        empty_like_dt = torch.empty_like(dist_tensor)
        # empty is not deterministic, so we only check that the shard propagation worked
        self.assertEqual((4, 8), empty_like_dt.to_local().shape)

    @with_comms
    def test_fill_inplace(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.fill_(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())
        self.assertEqual(full_expected, dist_tensor.to_local())

    @with_comms
    def test_full_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.full_like(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())

    @with_comms
    def test_ones_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(4, 8)
        self.assertEqual(ones_expected, ones_like_dt.to_local())

    @with_comms
    def test_zero_inplace(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zero_(dist_tensor)
        zeros_expected = torch.zeros(4, 8)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        self.assertEqual(zeros_expected, dist_tensor.to_local())

    @with_comms
    def test_zeros_like(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zeros_like(dist_tensor)
        zeros_expected = torch.zeros(4, 8)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())

    @with_comms
    def test_softmax_fwd(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        x = torch.rand(8, 12, 16, device=self.device_type)

        dims = range(3)  # used to convert -1 to the actual dim
        for softmax_dim in range(-1, 3):
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            )
            for batch_dim in range(-1, 3):
                dist_x = distribute_tensor(x, device_mesh, [Shard(batch_dim)])
                dist_y = torch.nn.functional.softmax(
                    dist_x, dim=softmax_dim, dtype=torch.float32
                )
                if dims[batch_dim] == dims[softmax_dim]:
                    # in this case the tensor is implicitly replicated
                    self.assertTrue(dist_y.placements[0].is_replicate())
                else:
                    self.assertTrue(
                        dist_y.placements[0].is_shard(dim=batch_dim)
                    )
                dist_y = dist_y.redistribute(device_mesh, [Replicate()])
                self.assertEqual(dist_y.to_local(), local_y)

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_softmax_cpu_gpu_discrepancy(self):
        softmax_dims = [-1, 0, 1, 2]
        # torch.tensor
        for softmax_dim in softmax_dims:
            tensor_cpu = torch.rand(8, 12, 16, device="cpu", requires_grad=True)
            tensor_gpu = tensor_cpu.clone().detach().requires_grad_(True).cuda()
            res_cpu = torch.nn.functional.softmax(
                tensor_cpu, dim=softmax_dim, dtype=torch.float32
            )
            res_gpu = torch.nn.functional.softmax(
                tensor_gpu, dim=softmax_dim, dtype=torch.float32
            )
            self.assertEqual(res_cpu, res_gpu.to(device="cpu"))
            res_cpu.sum().backward()
            res_gpu.sum().backward()
            self.assertIsNotNone(tensor_cpu.grad)
            self.assertIsNotNone(tensor_gpu.grad)
            self.assertEqual(tensor_cpu.grad, tensor_gpu.grad.to(device="cpu"))

        # DTensor
        shard_dims = [-1, 0, 1, 2]
        args_list = list(
            itertools.product(softmax_dims, shard_dims)
        )
        fail_list = []
        cpu_mesh = DeviceMesh("cpu", list(range(self.world_size)))
        gpu_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        for softmax_dim, shard_dim in args_list:
            if (softmax_dim, shard_dim) not in fail_list:
                tensor_cpu = torch.rand(8, 12, 16, device="cpu", requires_grad=True)
                tensor_gpu = tensor_cpu.clone().detach().requires_grad_(True).cuda()
                dist_x_cpu = distribute_tensor(tensor_cpu, cpu_mesh, [Shard(shard_dim)])
                dist_x_gpu = distribute_tensor(tensor_gpu, gpu_mesh, [Shard(shard_dim)])
                dist_y_cpu = tensor_cpu.softmax(dim=softmax_dim)
                dist_y_gpu = tensor_gpu.softmax(dim=softmax_dim)
                self.assertTrue(dist_y_cpu.to_local(), dist_y_gpu.to_local().to(device="cpu"))
                dist_y_cpu.sum().redistribute(cpu_mesh, [Replicate()]).backward()
                dist_y_gpu.sum().redistribute(gpu_mesh, [Replicate()]).backward()
                self.assertIsNotNone(dist_x_cpu.grad)
                self.assertIsNotNone(dist_x_gpu.grad)
                self.assertEqual(dist_x_cpu.grad.to_local(), dist_x_gpu.grad.to_local().to(device="cpu"))


    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_softmax_with_bwd(self):
        # test on CPU now has problem. See test_softmax_bwd_cpu_discrepancy
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        dims = range(3)  # used to convert -1 to the actual dim
        # failing cases: (0, -1), (1, -1)
        # for softmax_dim in range(-1, 3):
            # for batch_dim in range(0, 3):
        pass_list = [
            (-1, -1), # auto replicate
            (-1, 0),
            (-1, 1),
            (-1, 2), # auto replicate
            (0, 0), # auto replicate
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, -1), # auto replicate
            (2, 0),
            (2, 1),
            (2, 2), # auto replicate
        ]
        fail_list = [
            (0, -1),
            (1, -1),
        ]
        test_list = [
            (0, -1),
        ]
        for i in range(1):
            for softmax_dim, batch_dim in test_list:
                x = torch.rand(
                    4, 4, 4, device=self.device_type, requires_grad=True
                )
                self.assertTrue(x.requires_grad)
                local_y = torch.nn.functional.softmax(
                    x, dim=softmax_dim, dtype=torch.float32
                ).sum()
                local_y.backward()

                dist_x = distribute_tensor(x, device_mesh, [Shard(batch_dim)])
                self.assertTrue(dist_x.requires_grad)
                dist_softmax = dist_x.softmax(dim=softmax_dim)
                if dims[softmax_dim] == dims[batch_dim]:
                    # dtensor is suggested to replicate
                    self.assertTrue(dist_softmax.placements[0].is_replicate())
                else:
                    self.assertTrue(
                        dist_softmax.placements[0].is_shard(dim=batch_dim)
                    )
                dist_y = dist_softmax.sum()
                dist_y = dist_y.redistribute(device_mesh, [Replicate()])
                #print(f"dist sum={dist_y.to_local()}\nlocal sum={local_y}")
                self.assertEqual(dist_y.to_local(), local_y)
                self.assertIsNone(dist_x.grad)
                dist_y.backward()
                self.assertIsNotNone(dist_x.grad)
                dist_x_grad = dist_x.grad.redistribute(
                    device_mesh, [Replicate()]
                )
                print(f"dist_grad={dist_x_grad.to_local()}\nlocal_grad={x.grad}")
                #self.assertEqual(dist_x_grad.to_local(), x.grad)


if __name__ == "__main__":
    run_tests()
