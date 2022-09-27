# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, Tuple

from torch.distributed._shard.checkpoint.metadata import (
    STATE_DICT_TYPE,
)
"""
TODO
handle:
    tuple
    OrderedDict
    NamedTuple

change mappings from dict to a class
change set_element to recreate the right type (important with tuple, OD and ND)

"""

from .utils import (
    traverse_state_dict,
    set_element,
)

def flatten_state_dict(
    state_dict: STATE_DICT_TYPE
) -> Tuple[STATE_DICT_TYPE, Dict[str, Tuple]]:
    flattened = {}
    mappings = {}

    def flat_copy(path, value):
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings


def unflatten_state_dict(
    state_dict: STATE_DICT_TYPE,
    mapping: dict[str, Tuple]
) -> STATE_DICT_TYPE:
    inflated = {}
    for key, value in state_dict.items():
        set_element(inflated, mapping[key], value)
    return inflated

