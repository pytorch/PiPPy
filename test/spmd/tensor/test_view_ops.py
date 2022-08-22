# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import List, cast
from spmd.tensor.placement_types import Placement
from spmd.test._utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd import DeviceMesh, Shard, Replicate, distribute_tensor
from spmd.tensor.ops.view_ops import (
    ops,
    Singleton,
    Broadcast,
    Flatten,
    Repeat,
    Split,
    InputDim,
    view_groups,
)
from torch import Tensor, rand, randn
from torch.testing._internal.common_utils import run_tests
from torch.utils._pytree import tree_flatten

import itertools
import spmd
import torch
import torch.distributed as dist


class TestViewOps(DistTensorTestBase):
    def test_view_groups(self):
        self.assertEquals(
            view_groups([2, 3], [3, 2]),
            (
                Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
                Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
            ),
        )
        self.assertEquals(
            view_groups([3, 4, 5], [12, 5]),
            (
                Flatten((InputDim(0), InputDim(1))),
                InputDim(2),
            ),
        )
        self.assertEquals(
            view_groups([2, 3, 4, 5, 7], [12, 70]),
            (
                Split(
                    Flatten(
                        (
                            InputDim(0),
                            InputDim(1),
                            InputDim(2),
                            InputDim(3),
                            InputDim(4),
                        )
                    ),
                    (12, 70),
                    0,
                ),
                Split(
                    Flatten(
                        (
                            InputDim(0),
                            InputDim(1),
                            InputDim(2),
                            InputDim(3),
                            InputDim(4),
                        )
                    ),
                    (12, 70),
                    1,
                ),
            ),
        )
        self.assertEquals(
            view_groups([2, 3, 4, 5, 7], [3, 8, 7, 5]),
            (
                Split(
                    Flatten((InputDim(0), InputDim(1), InputDim(2))), (3, 8), 0
                ),
                Split(
                    Flatten((InputDim(0), InputDim(1), InputDim(2))), (3, 8), 1
                ),
                Split(Flatten((InputDim(3), InputDim(4))), (7, 5), 0),
                Split(Flatten((InputDim(3), InputDim(4))), (7, 5), 1),
            ),
        )
        self.assertEquals(
            view_groups([3, 4, 8, 3], [12, 4, 2, 3]),
            (
                Flatten((InputDim(0), InputDim(1))),
                Split(InputDim(2), (4, 2), 0),
                Split(InputDim(2), (4, 2), 1),
                InputDim(3),
            ),
        )
        self.assertEquals(
            view_groups([3, 24], [1, 3, 2, 4, 1, 3, 1]),
            (
                Singleton(),
                InputDim(0),
                Split(InputDim(1), (2, 4, 3), 0),
                Split(InputDim(1), (2, 4, 3), 1),
                Singleton(),
                Split(InputDim(1), (2, 4, 3), 2),
                Singleton(),
            ),
        )
        self.assertEquals(
            view_groups([1, 1, 3, 2, 1, 1], [6, 1, 1, 1]),
            (
                Flatten((InputDim(2), InputDim(3))),
                Singleton(),
                Singleton(),
                Singleton(),
            ),
        )
        self.assertEquals(
            view_groups([1, 1, 12, 1, 1, 1, 2, 5, 1], [3, 4, 1, 10]),
            (
                Split(InputDim(2), (3, 4), 0),
                Split(InputDim(2), (3, 4), 1),
                Singleton(),
                Flatten((InputDim(6), InputDim(7))),
            ),
        )
        self.assertEquals(
            view_groups([2, 3, 4], [2, -1, 4]),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

    @property
    def world_size(self) -> int:
        return 6

    def call_dt_test(self, op, args, kwargs, device_mesh: DeviceMesh):
        spmd.tensor.dispatch._DEBUG_STRICT = True
        spec = ops[op]
        rules = spec.dim_map(*args, **kwargs)
        outputs = op(*args, **kwargs)
        flat_args, _ = tree_flatten(args)
        in_shape = flat_args[0].shape

        no_shard_dims = set()
        for rule in rules:
            if isinstance(rule, Repeat):
                if isinstance(rule.input_dim, InputDim):
                    no_shard_dims.add(rule.input_dim.input_dim)
            elif isinstance(rule, Flatten):
                for dim in rule.input_dims[1:]:
                    if isinstance(dim, InputDim):
                        no_shard_dims.add(dim.input_dim)
            elif isinstance(rule, Split):
                if isinstance(rule.input_dim, Flatten):
                    for dim in rule.input_dim.input_dims[1:]:
                        if isinstance(dim, InputDim):
                            no_shard_dims.add(dim.input_dim)

        if op == torch.unbind:
            no_shard_dims.add(kwargs.get("dim", 0))

        sharding_choices = cast(List[Placement], [Replicate()]) + [
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
                self.assertEqual(outputs, full_out)

    def dimmap_test(self, op, args, expected_rule_output):
        rules = ops[op].dim_map(*args)
        self.assertEquals(rules, expected_rule_output)
        self.call_dt_test(op, args, {}, self.device_mesh)

    @with_comms
    def test_view_ops(self):
        self.device_mesh = DeviceMesh(
            self.device_type, torch.arange(dist.get_world_size()).view(-1, 2)
        )
        self.dimmap_test(torch.atleast_1d, (randn(()),), (Singleton(),))
        self.dimmap_test(torch.atleast_1d, (randn(24),), (InputDim(0),))
        self.dimmap_test(
            torch.atleast_1d, (randn(24, 36),), (InputDim(0), InputDim(1))
        )

        self.dimmap_test(
            torch.atleast_2d, (randn(()),), (Singleton(), Singleton())
        )
        self.dimmap_test(
            torch.atleast_2d, (randn(24),), (Singleton(), InputDim(0))
        )
        self.dimmap_test(
            torch.atleast_2d, (randn(24, 36),), (InputDim(0), InputDim(1))
        )
        self.dimmap_test(
            torch.atleast_2d,
            (randn(24, 36, 48),),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

        self.dimmap_test(
            torch.atleast_3d,
            (randn(()),),
            (Singleton(), Singleton(), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24),),
            (Singleton(), InputDim(0), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36),),
            (InputDim(0), InputDim(1), Singleton()),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36, 42),),
            (InputDim(0), InputDim(1), InputDim(2)),
        )
        self.dimmap_test(
            torch.atleast_3d,
            (randn(24, 36, 42, 24),),
            (InputDim(0), InputDim(1), InputDim(2), InputDim(3)),
        )

        with self.assertRaises(AssertionError):
            ops[torch.broadcast_to].dim_map(randn(24, 36), (1, 2, 4))

        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (1, 24, 36)),
            (Singleton(), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (42, 24, 36)),
            (Broadcast(Singleton(), 42), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 1, 36), (12, 24, 24, 36)),
            (
                Broadcast(Singleton(), 12),
                InputDim(0),
                Broadcast(InputDim(1), 24),
                InputDim(2),
            ),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 36), (-1, 36)),
            (InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.broadcast_to,
            (rand(24, 1, 36), (-1, 1, 36)),
            (InputDim(0), InputDim(1), InputDim(2)),
        )

        self.dimmap_test(
            torch.broadcast_to,
            (randn(36, 1, 24), (12, 36, 42, 24)),
            (
                Broadcast(Singleton(), 12),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
            ),
        )

        self.dimmap_test(
            Tensor.expand,
            (randn(24, 1, 36, 1), 36, 24, 42, -1, 24),
            (
                Broadcast(Singleton(), 36),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
                Broadcast(InputDim(3), 24),
            ),
        )

        self.dimmap_test(
            Tensor.expand,
            (randn(24, 1, 36, 1), (36, 24, 42, -1, 24)),
            (
                Broadcast(Singleton(), 36),
                InputDim(0),
                Broadcast(InputDim(1), 42),
                InputDim(2),
                Broadcast(InputDim(3), 24),
            ),
        )

        self.dimmap_test(
            torch.flatten,
            (randn(24, 36),),
            (Flatten((InputDim(0), InputDim(1))),),
        )
        self.dimmap_test(torch.flatten, (randn(42),), (InputDim(0),))
        self.dimmap_test(torch.flatten, (randn(()),), (Singleton(),))

        self.dimmap_test(
            torch.movedim,
            (randn(12, 24, 48, 96), 1, 2),
            (InputDim(0), InputDim(2), InputDim(1), InputDim(3)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(6, 12, 24), 1, 0),
            (InputDim(1), InputDim(0), InputDim(2)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 12, 6), (1, 2), (0, 1)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 6, 12), (0, 2, 1), (2, 1, 0)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(24, 12), (1, 0), (0, 1)),
            (InputDim(1), InputDim(0)),
        )

        self.dimmap_test(
            torch.movedim,
            (randn(36, 24, 12), (1, 2), (0, 1)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )
        self.dimmap_test(
            torch.movedim,
            (randn(36, 24, 12), (1, 2), (-3, -2)),
            (InputDim(1), InputDim(2), InputDim(0)),
        )

        self.dimmap_test(
            torch.permute,
            (randn(24, 36, 42), (2, 0, 1)),
            (InputDim(2), InputDim(0), InputDim(1)),
        )
        self.dimmap_test(
            torch.permute,
            (randn(24, 36, 42), (-1, -3, -2)),
            (InputDim(2), InputDim(0), InputDim(1)),
        )

        self.dimmap_test(
            torch.ravel,
            (randn(24, 36),),
            (Flatten((InputDim(0), InputDim(1))),),
        )
        self.dimmap_test(torch.ravel, (randn(42),), (InputDim(0),))
        self.dimmap_test(torch.ravel, (randn(()),), (Singleton(),))

        self.dimmap_test(
            Tensor.repeat,
            (randn(24, 36), 1, 2, 1, 1, 2),
            (
                Singleton(),
                Broadcast(Singleton(), 2),
                Singleton(),
                InputDim(0),
                Repeat(InputDim(1), 2),
            ),
        )

        self.dimmap_test(
            torch.reshape,
            (randn(6, 12, 24), (72, 24)),
            (Flatten((InputDim(0), InputDim(1))), InputDim(2)),
        )

        self.dimmap_test(
            torch.tile,
            (randn(24, 36), (1, 2, 1, 1, 2)),
            (
                Singleton(),
                Broadcast(Singleton(), 2),
                Singleton(),
                InputDim(0),
                Repeat(InputDim(1), 2),
            ),
        )
        self.dimmap_test(
            torch.tile,
            (
                randn(42, 24, 36),
                (
                    1,
                    3,
                ),
            ),
            (InputDim(0), InputDim(1), Repeat(InputDim(2), 3)),
        )

        self.dimmap_test(
            torch.transpose,
            (randn(24, 60, 42, 60), 2, 0),
            (InputDim(2), InputDim(1), InputDim(0), InputDim(3)),
        )
        self.dimmap_test(
            torch.transpose,
            (randn(24, 60, 42, 60), -1, 0),
            (InputDim(3), InputDim(1), InputDim(2), InputDim(0)),
        )

        self.dimmap_test(
            torch.unsqueeze,
            (randn(42, 24, 36), 1),
            (InputDim(0), Singleton(), InputDim(1), InputDim(2)),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(6, 12, 24), 72, 24),
            (Flatten((InputDim(0), InputDim(1))), InputDim(2)),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(1, 1, 12), -1),
            (InputDim(2),),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(1, 1, 42, 24), -1),
            (Flatten((InputDim(2), InputDim(3))),),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(1, 1, 42, 1, 24, 1), -1),
            (Flatten((InputDim(2), InputDim(4))),),
        )

        self.dimmap_test(
            Tensor.view,
            (randn(48, 35, 26), (24, 4, 35, 13)),
            (
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=0,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=1,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=2,
                ),
                Split(
                    Flatten(input_dims=(InputDim(0), InputDim(1), InputDim(2))),
                    group_shape=(24, 4, 35, 13),
                    split_id=3,
                ),
            ),
        )


if __name__ == "__main__":
    run_tests()
