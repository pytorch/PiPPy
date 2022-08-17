# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.placement_types import Replicate
import torch
from torch.testing._internal.common_utils import run_tests
from spmd.test._utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd import DeviceMesh, Shard, Replicate, distribute_tensor

from torch.utils._pytree import tree_flatten

import spmd
from spmd.tensor.ops.view_ops import SINGLETON, BROADCAST, FLATTEN, REPEAT, ops

import itertools

import torch.distributed as dist
from torch import Tensor


def call_dt_test(op, args, kwargs, device_mesh: DeviceMesh):
    spmd.tensor.dispatch._DEBUG_STRICT = True
    spec = ops[op]
    rules = spec.dim_map(*args, **kwargs)
    outputs = op(*args, **kwargs)
    flat_args, _ = tree_flatten(args)
    in_shape = flat_args[0].shape

    assert not isinstance(rules, list)

    if dist.get_rank() == 0:
        print("-------- ", op)

    no_shard_dims = set()
    for rule in rules:
        if isinstance(rule, tuple):
            if rule[0] == "repeat":
                no_shard_dims.add(rule[1])
            if rule[0] == "flatten":
                no_shard_dims |= set(rule[1][1:])
            if rule[0] == "comp":
                if isinstance(rule[1], tuple) and rule[1][0] == "flatten":
                    no_shard_dims |= set(rule[1][1][1:])

    if op == torch.unbind:
        no_shard_dims.add(kwargs.get("dim", 0))

    sharding_choices = [Replicate()] + [
        Shard(i)
        for i, s in enumerate(in_shape)
        if s > 1 and i not in no_shard_dims
    ]

    all_sharding_choices = itertools.product(
        *(device_mesh.ndim * [sharding_choices])
    )

    for in_shard in all_sharding_choices:
        # print(f'   |--- {in_shard}')
        in_dt = distribute_tensor(args[0], device_mesh, in_shard)
        out_dt = op(in_dt, *args[1:], **kwargs)

        full_out = out_dt.redistribute(
            device_mesh, device_mesh.ndim * [Replicate()]
        ).to_local()

        if dist.get_rank() == 0:
            assert (
                outputs.shape == full_out.shape
            ), f"Expected shape {outputs.shape}, got {full_out.shape}"
            assert torch.allclose(outputs, full_out)


def assert_throw(fn):
    try:
        fn()
    except:
        return
    assert False, "didnt throw"


from torch import rand, randn


DEVICE_MESH = None


def dimmap_test(op, args, expected_rule_output):
    assert ops[op].dim_map(*args) == expected_rule_output
    call_dt_test(op, args, {}, DEVICE_MESH)


def dimmap_tests():
    dimmap_test(torch.atleast_1d, (randn(()),), (SINGLETON,))
    dimmap_test(torch.atleast_1d, (randn(24),), (0,))
    dimmap_test(torch.atleast_1d, (randn(24, 36),), (0, 1))

    dimmap_test(torch.atleast_2d, (randn(()),), (SINGLETON, SINGLETON))
    dimmap_test(torch.atleast_2d, (randn(24),), (SINGLETON, 0))
    dimmap_test(torch.atleast_2d, (randn(24, 36),), (0, 1))
    dimmap_test(torch.atleast_2d, (randn(24, 36, 48),), (0, 1, 2))

    dimmap_test(
        torch.atleast_3d, (randn(()),), (SINGLETON, SINGLETON, SINGLETON)
    )
    dimmap_test(torch.atleast_3d, (randn(24),), (SINGLETON, 0, SINGLETON))
    dimmap_test(torch.atleast_3d, (randn(24, 36),), (0, 1, SINGLETON))
    dimmap_test(torch.atleast_3d, (randn(24, 36, 42),), (0, 1, 2))
    dimmap_test(torch.atleast_3d, (randn(24, 36, 42, 24),), (0, 1, 2, 3))

    assert_throw(
        lambda: ops[torch.broadcast_to].dim_map(randn(24, 36), (1, 2, 4))
    )

    dimmap_test(
        torch.broadcast_to, (rand(24, 36), (1, 24, 36)), (SINGLETON, 0, 1)
    )
    dimmap_test(
        torch.broadcast_to,
        (rand(24, 36), (42, 24, 36)),
        (BROADCAST(SINGLETON, 42), 0, 1),
    )
    dimmap_test(
        torch.broadcast_to,
        (rand(24, 1, 36), (12, 24, 24, 36)),
        (BROADCAST(SINGLETON, 12), 0, BROADCAST(1, 24), 2),
    )
    dimmap_test(torch.broadcast_to, (rand(24, 36), (-1, 36)), (0, 1))
    dimmap_test(torch.broadcast_to, (rand(24, 1, 36), (-1, 1, 36)), (0, 1, 2))

    dimmap_test(
        torch.broadcast_to,
        (randn(36, 1, 24), (12, 36, 42, 24)),
        (BROADCAST(SINGLETON, 12), 0, BROADCAST(1, 42), 2),
    )

    dimmap_test(
        Tensor.expand,
        (randn(24, 1, 36, 1), 36, 24, 42, -1, 24),
        (BROADCAST(SINGLETON, 36), 0, BROADCAST(1, 42), 2, BROADCAST(3, 24)),
    )

    dimmap_test(
        Tensor.expand,
        (randn(24, 1, 36, 1), (36, 24, 42, -1, 24)),
        (BROADCAST(SINGLETON, 36), 0, BROADCAST(1, 42), 2, BROADCAST(3, 24)),
    )

    dimmap_test(torch.flatten, (randn(24, 36),), (FLATTEN((0, 1)),))
    dimmap_test(torch.flatten, (randn(42),), (0,))
    dimmap_test(torch.flatten, (randn(()),), (SINGLETON,))

    dimmap_test(torch.movedim, (randn(12, 24, 48, 96), 1, 2), (0, 2, 1, 3))
    dimmap_test(torch.movedim, (randn(6, 12, 24), 1, 0), (1, 0, 2))
    dimmap_test(torch.movedim, (randn(24, 12, 6), (1, 2), (0, 1)), (1, 2, 0))
    dimmap_test(
        torch.movedim, (randn(24, 6, 12), (0, 2, 1), (2, 1, 0)), (1, 2, 0)
    )
    dimmap_test(torch.movedim, (randn(24, 12), (1, 0), (0, 1)), (1, 0))

    dimmap_test(torch.movedim, (randn(36, 24, 12), (1, 2), (0, 1)), (1, 2, 0))

    dimmap_test(torch.permute, (randn(24, 36, 42), (2, 0, 1)), (2, 0, 1))

    dimmap_test(torch.ravel, (randn(24, 36),), (FLATTEN((0, 1)),))
    dimmap_test(torch.ravel, (randn(42),), (0,))
    dimmap_test(torch.ravel, (randn(()),), (SINGLETON,))

    dimmap_test(
        Tensor.repeat,
        (randn(24, 36), 1, 2, 1, 1, 2),
        (SINGLETON, BROADCAST(SINGLETON, 2), SINGLETON, 0, REPEAT(1, 2)),
    )

    dimmap_test(
        torch.reshape,
        (randn(6, 12, 24), (72, 24)),
        (FLATTEN((0, 1)), 2),
    )

    dimmap_test(
        torch.tile,
        (randn(24, 36), (1, 2, 1, 1, 2)),
        (SINGLETON, BROADCAST(SINGLETON, 2), SINGLETON, 0, REPEAT(1, 2)),
    )
    dimmap_test(
        torch.tile,
        (
            randn(42, 24, 36),
            (
                1,
                3,
            ),
        ),
        (0, 1, REPEAT(2, 3)),
    )

    dimmap_test(torch.transpose, (randn(24, 60, 42, 60), 2, 0), (2, 1, 0, 3))

    dimmap_test(torch.unsqueeze, (randn(42, 24, 36), 1), (0, SINGLETON, 1, 2))

    dimmap_test(
        Tensor.view,
        (randn(6, 12, 24), 72, 24),
        (FLATTEN((0, 1)), 2),
    )

    dimmap_test(
        Tensor.view,
        (randn(1, 1, 12), -1),
        (2,),
    )

    dimmap_test(
        Tensor.view,
        (randn(1, 1, 42, 24), -1),
        (FLATTEN((2, 3)),),
    )

    dimmap_test(
        Tensor.view,
        (randn(1, 1, 42, 1, 24, 1), -1),
        (FLATTEN((2, 4)),),
    )


class TestViewOps(DistTensorTestBase):
    @property
    def world_size(self) -> int:
        return 6

    @with_comms
    def test_view_ops(self):
        global DEVICE_MESH
        DEVICE_MESH = DeviceMesh(
            "cpu", torch.arange(dist.get_world_size()).view(-1, 2)
        )
        dimmap_tests()


if __name__ == "__main__":
    run_tests()
