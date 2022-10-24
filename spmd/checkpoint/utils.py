# Copyright (c) Meta Platforms, Inc. and affiliates

from typing import List, Sequence


def _element_wise_add(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [i_a + i_b for i_a, i_b in zip(a, b)]


def _element_wise_sub(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [i_a - i_b for i_a, i_b in zip(a, b)]
