<<<<<<< HEAD
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
=======
# Owner(s): ["oncall: distributed"]

import torch
from spmd.testing.common_dtensor import DTensorTestBase, with_comms
from spmd.tensor import distribute_tensor, DeviceMesh, Shard, Replicate
from spmd.tensor.parallel.style import (
    make_output_shard_1d,
    make_output_replicate_1d,
    make_output_tensor,
)


class TensorParallelStyleTest(DTensorTestBase):
    # Common logic for testing prepare output funcs
    def _test_prepare_output(self, func, spec, dim=None):
        device_mesh = DeviceMesh(self.device_type, [0, 1, 2, 3])
        tensor = torch.rand(8, 16, device=self.device_type)
        dtensor = distribute_tensor(tensor, device_mesh, spec)
        if dim is not None:
            output = func(dtensor, device_mesh, dim)
        else:
            output = func(dtensor, device_mesh)
        return output, dtensor, device_mesh

    @with_comms
    def test_make_output_shard_1d(self):
        # test when output is sharded.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_shard_1d, [Shard(0)], 1
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(1)]))
        #  test when output is replicated.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_shard_1d, [Replicate()], 0
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]))

    @with_comms
    def test_make_output_replicate_1d(self):
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_replicate_1d, [Shard(0)]
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()])
        )

    @with_comms
    def test_make_output_tensor(self):
        # test when output is sharded.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_tensor, [Shard(0)]
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()]).to_local()
        )
        #  test when output is replicated.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_tensor, [Replicate()]
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()]).to_local()
        )

    # Common logic for testing prepare output funcs errors.
    def _test_prepare_output_error(self, func):
        tensor = torch.rand(8, 16, device=self.device_type)
        device_mesh = DeviceMesh(self.device_type, [0, 1, 2, 3])
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        output = [dtensor]
        with self.assertRaisesRegex(
            AssertionError,
            f"output of Tensor Parallel is actually {type(output)} not DTensor.",
        ):
            func(output, device_mesh)
        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        with self.assertRaisesRegex(
            AssertionError, f"{func.__name__}: device mesh is not 1D"
        ):
            func(dtensor, device_mesh)

    @with_comms
    def test_prepare_output_error(self):
        self._test_prepare_output_error(make_output_shard_1d)
        self._test_prepare_output_error(make_output_replicate_1d)
        self._test_prepare_output_error(make_output_tensor)
>>>>>>> origin/prepare_out
