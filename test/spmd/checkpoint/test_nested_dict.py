# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
import spmd.checkpoint.nested_dict as nd


class TestFlattening(TestCase):
    def test_flattening_round_trip(self) -> None:
        state_dict = {
            "key0": 1,
            "key1": [1, 2],
            "key2": {1: 2, 2: 3},
            "key3": torch.tensor([1]),
            "key4": [[torch.tensor(2), "x"], [1, 2, 3], {"key6": [44]}],
        }

        flatten_dict, mapping = nd.flatten_state_dict(state_dict)
        restored = nd.unflatten_state_dict(flatten_dict, mapping)

        self.assertEqual(state_dict, restored)

    def test_mapping(self) -> None:
        state_dict = {
            "k0": [1],
            "k2": [torch.tensor([1]), 99, [{"k3": torch.tensor(1)}]],
            "k3": ["x", 99, [{"k3": "y"}]],
        }

        flatten_dict, mapping = nd.flatten_state_dict(state_dict)
        self.assertIn(("k0",), mapping.values())
        self.assertIn(
            (
                "k2",
                0,
            ),
            mapping.values(),
        )
        self.assertIn(
            (
                "k2",
                1,
            ),
            mapping.values(),
        )
        self.assertIn(("k2", 2, 0, "k3"), mapping.values())
        self.assertIn(("k3",), mapping.values())


if __name__ == "__main__":
    run_tests()
