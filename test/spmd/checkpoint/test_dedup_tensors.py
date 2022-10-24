# Copyright (c) Meta Platforms, Inc. and affiliates

import dataclasses
import torch
from torch.distributed._shard.checkpoint.planner import SavePlan, WriteItemType
from torch.distributed._shard.checkpoint.planner_helpers import (
    _create_write_item_for_tensor,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from spmd.checkpoint import dedup_tensors


def create_plan(second_fqn) -> SavePlan:
    # the first wi is for a duplicated shard (that covers the whole tensor)
    wi1 = _create_write_item_for_tensor("tensor_0", torch.rand(4))
    wi1 = dataclasses.replace(wi1, type=WriteItemType.SHARD)

    # the second wi has different keys
    wi2 = _create_write_item_for_tensor(second_fqn, torch.rand(10))

    return SavePlan([wi1, wi2])


class TestDedupTensor(TestCase):
    def test_dedup_shards(self):
        rank0 = create_plan("r0")
        rank1 = create_plan("r1")

        dedup_plans = dedup_tensors([rank0, rank1])

        self.assertEqual(2, len(dedup_plans[0].items))
        self.assertEqual(1, len(dedup_plans[1].items))

        self.assertIn(
            "tensor_0", (item.index.fqn for item in dedup_plans[0].items)
        )
        self.assertIn("r0", (item.index.fqn for item in dedup_plans[0].items))

        self.assertIn("r1", (item.index.fqn for item in dedup_plans[1].items))


if __name__ == "__main__":
    run_tests()
