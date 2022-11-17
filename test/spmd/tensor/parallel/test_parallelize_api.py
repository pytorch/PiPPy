# Owner(s): ["oncall: distributed"]

import torch
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_dtensor import DTensorTestBase, with_comms
from spmd.tensor import distribute_tensor, DeviceMesh, Shard, Replicate
from spmd.tensor.parallel import PairwiseParallel
from spmd.tensor.parallel.api import _parallelize_mlp
from spmd.tensor.parallel.utils import _create_1d_device_mesh


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
        one_dimention_mesh_shape = mesh_shape[self.rank // tp_size, :]
        pg = mesh.get_dim_groups()[1]
        new_mesh = _create_1d_device_mesh(mesh, 1)
        expected_mesh = DeviceMesh(
            self.device_type, one_dimention_mesh_shape, [pg]
        )
        self.assertEqual(new_mesh.mesh, expected_mesh.mesh)
        self.assertEqual(new_mesh.device_type, expected_mesh.device_type)

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
        _parallelize_mlp(model_tp, PairwiseParallel(), device_mesh)

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


if __name__ == "__main__":
    run_tests()
