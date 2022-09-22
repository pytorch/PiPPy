# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Sequence, Any, Dict, Callable, Optional

import torch
from torch import Tensor
from torch.testing._internal.common_utils import run_tests
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
    TEST_GPU_NUM,
)
from spmd.tensor.dispatch import OpSchema

from spmd.tensor.ops.pointwise_ops import pointwise_rule
from spmd import DeviceMesh, DTensor
from spmd.tensor.placement_types import (
    Shard,
    Replicate,
    _Partial,
    DTensorSpec,
    Placement,
)
from torch.distributed.distributed_c10d import ReduceOp
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu


def deepcopy_convert_to_dtensor(
    val: Any,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Any:
    """
    Recursively coerce (over Sequence and Dict types) Tensors into DTensors.

    :param device_mesh: the DeviceMesh to use.
    :param placements: the Placement list to use.
    :return: the transformed structure.
    """
    if isinstance(val, Tensor):
        return DTensor(
            val,
            device_mesh=device_mesh,
            placements=placements,
            requires_grad=val.requires_grad,
        )

    if isinstance(val, Sequence):
        return [
            deepcopy_convert_to_dtensor(
                x,
                device_mesh=device_mesh,
                placements=placements,
            )
            for x in val
        ]

    if isinstance(val, Dict):
        return {
            k: deepcopy_convert_to_dtensor(
                v,
                device_mesh=device_mesh,
                placements=placements,
            )
            for k, v in val.items()
        }

    return val


def deepcopy_convert_from_dtensor(val: Any) -> Any:
    """
    Recursive coerce any DTensor to local Tensor.

    :param val: the structure to coerce.
    :return: the coerced structure.
    """
    if isinstance(val, DTensor):
        return val.to_local()

    if isinstance(val, Sequence):
        return [deepcopy_convert_from_dtensor(x) for x in val]

    if isinstance(val, Dict):
        return {k: deepcopy_convert_from_dtensor(v) for k, v in val.items()}

    return val


class DistElementwiseOpsTest(DistTensorTestBase):
    def _compare_pairwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        op: Callable,
        reset_seed: Optional[Callable] = None,
        args: Sequence[Any] = tuple(),
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not kwargs:
            kwargs = {}

        torch.manual_seed(self.rank)

        dargs = deepcopy_convert_to_dtensor(
            args,
            device_mesh=device_mesh,
            placements=placements,
        )
        dkwargs = deepcopy_convert_to_dtensor(
            kwargs,
            device_mesh=device_mesh,
            placements=placements,
        )

        reset_seed() if reset_seed else None

        # run the reference first, in case the call is broken;
        # it's better to debug an incorrect call at this point.
        reference_result = op(*args, **kwargs)

        reset_seed() if reset_seed else None

        dist_result = op(*dargs, **dkwargs)

        collected_result = deepcopy_convert_from_dtensor(dist_result)

        self.assertEqual(reference_result, collected_result)

    # TODO: We need to add CPU tests for ops in the future.
    def _run_sharded_elementwise_ops(
        self,
        *,
        mesh,
        spec,
        input_size,
        op,
        reset_seed=None,
        **kwargs,
    ):
        input_tensor = torch.randn(
            *input_size, device=self.device_type, requires_grad=True
        )

        self._compare_pairwise_ops(
            device_mesh=mesh,
            placements=spec,
            reset_seed=reset_seed,
            op=op,
            args=(input_tensor,),
            kwargs=kwargs,
        )

    @with_comms
    def test_activations(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self._run_sharded_elementwise_ops(
            mesh=device_mesh,
            spec=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            mesh=device_mesh,
            spec=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            mesh=device_mesh,
            spec=[Shard(1)],
            input_size=(3, 14),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            mesh=device_mesh,
            spec=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            mesh=device_mesh,
            spec=[Shard(0)],
            input_size=(8, 5),
            op=torch.sigmoid,
        )
        self._run_sharded_elementwise_ops(
            mesh=device_mesh,
            spec=[Replicate()],
            input_size=(8, 5),
            op=torch.sigmoid,
        )

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_dropout(self):
        def _reset_random_seed():
            torch.manual_seed(self.rank + 4)

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self._run_sharded_elementwise_ops(
            mesh=device_mesh,
            spec=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.dropout,
            p=0.4,
            training=False,
        )
        self._run_sharded_elementwise_ops(
            mesh=device_mesh,
            spec=[Shard(1)],
            input_size=(3, 14),
            op=torch.nn.functional.dropout,
            reset_seed=_reset_random_seed,
            p=0.5,
            training=True,
        )

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_dropout_backward(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        placements = [Shard(0)]

        input_size = (8, 5)

        grad_output = torch.rand(
            input_size,
            device=self.device_type,
            requires_grad=True,
        )
        mask = (
            torch.rand(
                input_size,
                device=self.device_type,
                requires_grad=False,
            )
            < 0.8
        )

        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            op=torch.ops.aten.native_dropout_backward,
            kwargs=dict(
                grad_output=grad_output,
                mask=mask,
                scale=0.3,
            ),
        )

    @with_comms
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    def test_dropout_errors(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        with self.assertRaisesRegex(RuntimeError, "Not supported!"):
            self._run_sharded_elementwise_ops(
                mesh=device_mesh,
                spec=[_Partial(ReduceOp.SUM)],
                input_size=(8, 5),
                op=torch.nn.functional.dropout,
            )

    @with_comms
    def test_mul_out(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        shard_spec = [Shard(0)]
        input_size = (8, 4)
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)

        other_tensor = torch.randn(*input_size, device=self.device_type)
        other_dtensor = DTensor.from_local(other_tensor, device_mesh, shard_spec)

        output_tensor = torch.randn(*input_size, device=self.device_type)
        output_dtensor = DTensor.from_local(output_tensor, device_mesh, shard_spec)
        dt = torch.mul(dtensor, other_dtensor, out=output_dtensor)
        expected = torch.mul(input_tensor, other_tensor, out=output_tensor)
        self.assertEqual(input_tensor, dtensor.to_local())
        self.assertEqual(expected, dt.to_local())

    @with_comms
    def test_pointwise_rules_suggestion(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # propagate point-wise sharding
        inp1, inp2 = [-1, -1], [-1, 0]
        mat1_spec = DTensorSpec.from_dim_map(mesh, inp1, [], shape=torch.Size([8, 4]))
        mat2_spec = DTensorSpec.from_dim_map(mesh, inp2, [], shape=torch.Size([8, 4]))
        # adding a positional argument -1 to arg schema
        output_sharding = pointwise_rule(OpSchema((mat1_spec, mat2_spec, -1), {}))
        self.assertIsNone(output_sharding.output_spec)
        self.assertIsNotNone(output_sharding.schema_suggestions)

        # ensure that the suggestion from pointwise rules still have
        # the positional args that are not DTensorSpec
        schema_suggestion = output_sharding.schema_suggestions[0]
        self.assertEqual(len(schema_suggestion.args_schema), 3)
        self.assertEqual(schema_suggestion.args_schema[2], -1)


if __name__ == "__main__":
    run_tests()
