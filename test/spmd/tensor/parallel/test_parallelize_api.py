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
    @with_comms
    def test_creat_1d_device_mesh(self):
        tp_size = 2
        mesh_shape = (
            torch.arange(self.world_size)
            .reshape(
                self.world_size // (self.world_size // tp_size),
                self.world_size // tp_size,
            )
            .to(torch.int)
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)
        # When 1D dim is 1.
        one_dimention_mesh_shape = mesh_shape[self.rank // tp_size, :]
        pg = mesh.get_dim_groups()[1]
        new_mesh = _create_1d_device_mesh(mesh, 1)
        expected_mesh = DeviceMesh(
            self.device_type, one_dimention_mesh_shape, [pg]
        )
        self.assertEqual(new_mesh.mesh, expected_mesh.mesh)
        self.assertEqual(new_mesh.device_type, expected_mesh.device_type)
        # When 1D dim is 0.
        one_dimention_mesh_shape = mesh_shape[:, self.rank % tp_size]
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
        torch.manual_seed(self.rank)
        inp = torch.rand(*inp_size, device=self.device_type)
        rowwise = RowwiseParallel()

        torch.manual_seed(5)
        model = nn.Linear(16, 10, device=self.device_type)
        torch.manual_seed(5)
        model_tp = nn.Linear(16, 10, device=self.device_type)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(NUM_DEVICES)))
        _parallelize_linear(model_tp, device_mesh, rowwise)

        self.assertEqual(
            distribute_tensor(model.weight, device_mesh, [Shard(1)]).to_local(),
            model_tp.weight.to_local(),
        )
        self.assertEqual(
            distribute_tensor(
                model.bias, device_mesh, [Replicate()]
            ).to_local(),
            model_tp.bias.to_local(),
        )

        LR = 0.25
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        local_inp = make_input_replicate_1d(
            DTensor.from_local(inp, device_mesh, [Shard(0)])
        ).to_local()
        output = model(local_inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp.to_local())

        output.sum().backward()
        output_tp.sum().backward()

        replicate = [Replicate()]
        # Ensure gradients are same.
        self.assertEqual(
            model.weight.grad,
            model_tp.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.bias.grad,
            model_tp.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        optim.step()
        optim_tp.step()

        # Ensure params are same.
        self.assertEqual(
            model.weight,
            model_tp.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.bias,
            model_tp.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        inp = torch.rand(*inp_size, device=self.device_type)
        local_inp = make_input_replicate_1d(
            DTensor.from_local(inp, device_mesh, [Shard(0)])
        ).to_local()
        output = model(local_inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp.to_local())

    @with_comms
    def test_linear_col_wise_parallel(self):
        # test ColwiseParallel
        inp_size = [8, 10]
        inp = torch.rand(*inp_size, device=self.device_type)
        colwise = ColwiseParallel()

        torch.manual_seed(5)
        model = nn.Linear(10, 16, device=self.device_type)
        torch.manual_seed(5)
        model_tp = nn.Linear(10, 16, device=self.device_type)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(NUM_DEVICES)))
        _parallelize_linear(model_tp, device_mesh, colwise)

        self.assertEqual(
            distribute_tensor(model.weight, device_mesh, [Shard(0)]).to_local(),
            model_tp.weight.to_local(),
        )
        self.assertEqual(
            distribute_tensor(model.bias, device_mesh, [Shard(0)]).to_local(),
            model_tp.bias.to_local(),
        )

        LR = 0.25
        optim = torch.optim.SGD(model.parameters(), lr=LR)
        optim_tp = torch.optim.SGD(model_tp.parameters(), lr=LR)

        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp.to_local())

        output.sum().backward()
        output_tp.sum().backward()

        replicate = [Replicate()]
        # Ensure gradients are same.
        self.assertEqual(
            model.weight.grad,
            model_tp.weight.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.bias.grad,
            model_tp.bias.grad.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        optim.step()
        optim_tp.step()

        # Ensure params are same.
        self.assertEqual(
            model.weight,
            model_tp.weight.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )
        self.assertEqual(
            model.bias,
            model_tp.bias.redistribute(
                device_mesh=device_mesh, placements=replicate
            ).to_local(),
        )

        inp = torch.rand(*inp_size, device=self.device_type)
        output = model(inp)
        output_tp = model_tp(inp)
        self.assertEqual(output, output_tp.to_local())


if __name__ == "__main__":
    run_tests()
