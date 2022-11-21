# Owner(s): ["oncall: distributed"]

import torch
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_dtensor import DTensorTestBase, with_comms
from spmd.tensor import distribute_tensor, DeviceMesh, Shard, Replicate, DTensor
from spmd.tensor.parallel import (
    ParallelStyle,
    RowwiseParallel,
    ColwiseParallel,
    PairwiseParallel,
)
from spmd.tensor.parallel.api import (
    _parallelize_linear,
    _parallelize_mlp,
)
from spmd.tensor.parallel.utils import _create_1d_device_mesh
from spmd.tensor.parallel.style import (
    make_input_replicate_1d,
    make_output_replicate_1d,
)


class MLPModule(torch.nn.Module):
    def __init__(self, device):
        super(MLPModule, self).__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(10, 16, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 12, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class TensorParallelAPITests(DTensorTestBase):
    @property
    def world_size(self):
        gpu_num = torch.cuda.device_count()
        return gpu_num if gpu_num % 2 == 0 and gpu_num > 4 else 4

    @with_comms
    def test_creat_1d_device_mesh(self):
        dim_one_size = 2
        mesh_shape = (
            torch.arange(self.world_size)
            .reshape(
                self.world_size // dim_one_size,
                dim_one_size,
            )
            .to(torch.int)
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)
        # When 1D dim is 1.
        one_dimention_mesh_shape = mesh_shape[self.rank // dim_one_size, :]
        pg = mesh.get_dim_groups()[1]
        new_mesh = _create_1d_device_mesh(mesh, 1)
        expected_mesh = DeviceMesh(
            self.device_type, one_dimention_mesh_shape, [pg]
        )
        self.assertEqual(new_mesh.mesh, expected_mesh.mesh)
        self.assertEqual(new_mesh.device_type, expected_mesh.device_type)
        # When 1D dim is 0.
        one_dimention_mesh_shape = mesh_shape[:, self.rank % dim_one_size]
        pg = mesh.get_dim_groups()[0]
        new_mesh = _create_1d_device_mesh(mesh, 0)
        expected_mesh = DeviceMesh(
            self.device_type, one_dimention_mesh_shape, [pg]
        )
        self.assertEqual(new_mesh.mesh, expected_mesh.mesh)
        self.assertEqual(new_mesh.device_type, expected_mesh.device_type)

    @with_comms
    def test_creat_1d_device_mesh_error(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        with self.assertRaisesRegex(
            AssertionError,
            "Expect tp_mesh_dim within range \\[-1, 1\\), but found 3.",
        ):
            _create_1d_device_mesh(mesh, 3)

    def _check_module(
        self,
        local_module,
        dist_module,
        local_input,
        dist_input,
        device_mesh,
        local_optim,
        dist_optim,
    ):
        replicate = [Replicate()]

        # ensure the parameter is properly distributed.
        # TODO: see how to make it work on `mlp`: nested modules
        for name, param in local_module.named_parameters():
            dist_param = dist_module.get_parameter(name)
            self.assertEqual(
                param,
                dist_param.redistribute(
                    device_mesh=device_mesh, placements=replicate
                ).to_local(),
            )

        # clear gradients
        local_optim.zero_grad()
        dist_optim.zero_grad()

        # check forward correctness
        local_output = local_module(local_input)
        dist_output = dist_module(dist_input)
        self.assertEqual(local_output, dist_output.to_local())

        local_output.sum().backward()
        dist_output.sum().backward()

        # check backward correctness:
        #   ensure gradients are same
        for name, param in local_module.named_parameters():
            dist_param = dist_module.get_parameter(name)
            self.assertEqual(
                param.grad,
                dist_param.grad.redistribute(
                    device_mesh=device_mesh, placements=replicate
                ).to_local(),
            )

        local_optim.step()
        dist_optim.step()

    @with_comms
    def test_parallelize_mlp(self):
        model = MLPModule(self.device_type)
        model_tp = MLPModule(self.device_type)

        # Ensure model are initialized the same way.
        self.assertEqual(model.net1.weight, model_tp.net1.weight)
        self.assertEqual(model.net1.bias, model_tp.net1.bias)
        self.assertEqual(model.net2.weight, model_tp.net2.weight)
        self.assertEqual(model.net2.bias, model_tp.net2.bias)

        # Parallelize module.
        device_mesh = DeviceMesh(
            self.device_type, torch.arange(self.world_size)
        )
        _parallelize_mlp(model_tp, device_mesh, PairwiseParallel())

        # Ensure the parameter is properly distributed.
        self.assertEqual(
            distribute_tensor(model.net1.weight, device_mesh, [Shard(0)]),
            model_tp.net1.weight,
        )
        self.assertEqual(
            distribute_tensor(model.net1.bias, device_mesh, [Shard(0)]),
            model_tp.net1.bias,
        )
        self.assertEqual(
            distribute_tensor(model.net2.weight, device_mesh, [Shard(1)]),
            model_tp.net2.weight,
        )
        self.assertEqual(
            distribute_tensor(model.net2.bias, device_mesh, [Replicate()]),
            model_tp.net2.bias,
        )

    @with_comms
    def test_parallelize_mlp_error(self):
        class DummyParallel(ParallelStyle):
            def __init__(self) -> None:
                super().__init__(
                    make_input_replicate_1d, make_output_replicate_1d
                )

        model_tp = MLPModule(self.device_type)
        device_mesh = DeviceMesh(
            self.device_type, torch.arange(self.world_size)
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "Only support PairwiseParallel for MLP parallelization.",
        ):
            _parallelize_mlp(model_tp, device_mesh, DummyParallel())

        with self.assertRaisesRegex(
            RuntimeError, "We only support even number of Linear for MLP."
        ):
            _parallelize_mlp(
                torch.nn.Linear(10, 5), device_mesh, PairwiseParallel()
            )

    @with_comms
    def test_linear_row_wise_parallel(self):
        # test RowwiseParallel
        inp_size = [8, 16]
        rowwise = RowwiseParallel()

        torch.manual_seed(5)
        model = torch.nn.Linear(16, 10, device=self.device_type)
        torch.manual_seed(5)
        model_tp = torch.nn.Linear(16, 10, device=self.device_type)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        _parallelize_linear(model_tp, device_mesh, rowwise)

        LR = 0.25  # the learning rate we use for testing
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        torch.manual_seed(self.rank)
        for _ in range(2):
            inp = torch.rand(*inp_size, device=self.device_type)
            local_inp = make_input_replicate_1d(
                DTensor.from_local(
                    inp, device_mesh, [Shard(0)], run_check=False
                )
            ).to_local()
            self._check_module(
                model, model_tp, local_inp, inp, device_mesh, optim, optim_tp
            )

    @with_comms
    def test_linear_col_wise_parallel(self):
        # test ColwiseParallel
        inp_size = [8, 10]
        colwise = ColwiseParallel()

        torch.manual_seed(5)
        model = torch.nn.Linear(10, 16, device=self.device_type)
        torch.manual_seed(5)
        model_tp = torch.nn.Linear(10, 16, device=self.device_type)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        _parallelize_linear(model_tp, device_mesh, colwise)

        LR = 0.25  # the learning rate we use for testing
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        for _ in range(2):
            inp = torch.rand(*inp_size, device=self.device_type)
            self._check_module(
                model, model_tp, inp, inp, device_mesh, optim, optim_tp
            )


if __name__ == "__main__":
    run_tests()
