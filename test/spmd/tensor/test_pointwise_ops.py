# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from spmd.test._utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
    TEST_GPU_NUM,
)
from spmd import DeviceMesh, DTensor, distribute_tensor
from spmd.tensor.placement_types import Shard, Replicate, _Partial
from torch.distributed.distributed_c10d import ReduceOp
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu


class DistElementwiseOpsTest(DistTensorTestBase):
    # TODO: We need to add CPU tests for ops in the future.
    def _run_sharded_elementwise_ops(
        self, mesh, spec, input_size, op, reset_seed=None, **kwargs
    ):
        torch.manual_seed(self.rank)
        input_tensor = torch.randn(
            *input_size, device=self.device_type, requires_grad=True
        )
        dist_tensor = DTensor(
            input_tensor, mesh, spec, requires_grad=input_tensor.requires_grad
        )
        reset_seed() if reset_seed else None
        dt = op(dist_tensor, **kwargs)
        reset_seed() if reset_seed else None
        expected = op(input_tensor, **kwargs)
        self.assertEqual(input_tensor, dist_tensor.to_local())
        self.assertEqual(expected, dt.to_local())

    @with_comms
    def test_activations(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self._run_sharded_elementwise_ops(
            device_mesh, [Shard(0)], (8, 5), torch.nn.functional.gelu
        )
        self._run_sharded_elementwise_ops(
            device_mesh, [Replicate()], (8, 5), torch.nn.functional.gelu
        )
        self._run_sharded_elementwise_ops(
            device_mesh, [Shard(1)], (3, 14), torch.nn.functional.relu
        )
        self._run_sharded_elementwise_ops(
            device_mesh, [Replicate()], (8, 5), torch.nn.functional.relu
        )
        self._run_sharded_elementwise_ops(
            device_mesh, [Shard(0)], (8, 5), torch.sigmoid
        )
        self._run_sharded_elementwise_ops(
            device_mesh, [Replicate()], (8, 5), torch.sigmoid
        )

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_dropout(self):
        def _reset_random_seed():
            torch.manual_seed(self.rank + 4)

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self._run_sharded_elementwise_ops(
            device_mesh,
            [Shard(0)],
            (8, 5),
            torch.nn.functional.dropout,
            p=0.4,
            training=False,
        )
        self._run_sharded_elementwise_ops(
            device_mesh,
            [Shard(1)],
            (3, 14),
            torch.nn.functional.dropout,
            reset_seed=_reset_random_seed,
            p=0.5,
            training=True,
        )

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_dropout_errors(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        with self.assertRaisesRegex(RuntimeError, "Not supported!"):
            self._run_sharded_elementwise_ops(
                device_mesh,
                [_Partial(ReduceOp.SUM)],
                (8, 5),
                torch.nn.functional.dropout,
            )

    @with_comms
    def test_mul(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        spec = [Shard(0)]
        input_tensor = torch.randn(
            (8, 4), device=self.device_type, requires_grad=True
        )
        dtensor = DTensor.from_local(
            input_tensor, device_mesh, spec
        )
        # Mutiplication with a scalar.
        dt = torch.mul(dtensor, 8.0)
        expected = torch.mul(input_tensor, 8.0)
        self.assertEqual(input_tensor, dtensor.to_local())
        self.assertEqual(expected, dt.to_local())
        
        # Mutiplication with a matrix.
        input_tensor = torch.randn(
            (8, 4), device=self.device_type
        )
        dtensor = DTensor.from_local(
            input_tensor, device_mesh, spec
        )
        other_tensor = torch.randn(
            (8, 4), device=self.device_type
        )
        other_dtensor = DTensor.from_local(
            other_tensor, device_mesh, [Shard(0)]
        )
        output_tensor = torch.randn((8, 4), device=self.device_type)
        output_dtensor = DTensor.from_local(
            output_tensor, device_mesh, [Shard(0)]
        )
        dt = torch.mul(dtensor, other_dtensor, out=output_dtensor)
        expected = torch.mul(input_tensor, other_tensor, out=output_tensor)
        self.assertEqual(input_tensor, dtensor.to_local())
        self.assertEqual(expected, dt.to_local())


if __name__ == "__main__":
    run_tests()
