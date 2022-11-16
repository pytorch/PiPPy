# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import (
    DistTensorTestBase,
    with_comms,
    NUM_DEVICES,
)
from spmd.tensor import (
    DeviceMesh,
)
import spmd.tensor.parallel as tp


class DistTensorParallelExampleTest(DistTensorTestBase):
    @with_comms
    def test_make_input_replicate_1d(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        with self.assertRaisesRegex(
            AssertionError, "device_mesh is not passed nor can be inferred"
        ):
            dtensor = tp.style.make_input_replicate_1d(tensor)
        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        with self.assertRaisesRegex(
            AssertionError, "device_mesh dim is [0-9]+ but expcted to be 1"
        ):
            dtensor = tp.style.make_input_replicate_1d(tensor, device_mesh)

        device_mesh = DeviceMesh(self.device_type, list(range(NUM_DEVICES)))
        # test 1
        dtensor = tp.style.make_input_replicate_1d(tensor, device_mesh)
        self.assertEqual(tensor, dtensor.to_local())
        # test 2
        dtensor = tp.style.make_input_replicate_1d(dtensor)
        self.assertEqual(tensor, dtensor.to_local())
        # test 3
        dtensor = tp.style.make_input_replicate_1d(dtensor, device_mesh)
        self.assertEqual(tensor, dtensor.to_local())

    @with_comms
    def test_make_input_shard_1d(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        with self.assertRaisesRegex(
            AssertionError, "device_mesh is not passed nor can be inferred"
        ):
            dtensor = tp.style.make_input_shard_1d(tensor)
        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        with self.assertRaisesRegex(
            AssertionError, "device_mesh dim is [0-9]+ but expcted to be 1"
        ):
            dtensor = tp.style.make_input_shard_1d(tensor, device_mesh)

        device_mesh = DeviceMesh(self.device_type, list(range(NUM_DEVICES)))
        # test 1
        dtensor = tp.style.make_input_shard_1d(tensor, device_mesh)
        self.assertEqual(tensor, dtensor.to_local())
        # test 2
        dtensor = tp.style.make_input_shard_1d(dtensor)
        self.assertEqual(tensor, dtensor.to_local())
        # test 3
        dtensor = tp.style.make_input_shard_1d(dtensor, device_mesh)
        self.assertEqual(tensor, dtensor.to_local())


if __name__ == "__main__":
    run_tests()
