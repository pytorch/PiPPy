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

from .traverse import (
    traverse_state_dict,
    set_element,
    OBJ_PATH,
    STATE_DICT_ITEM,
)

FLATTEN_MAPPING = Dict[str, OBJ_PATH]
"""
Type for the flatenning metadata
"""


def flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.

    Returns:
        A tuple with the flaten state_dict and a mapping from original to new state_dict.

    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings


def unflatten_state_dict(
    state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING
) -> STATE_DICT_TYPE:
    """
    Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``
    """
    inflated: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        set_element(inflated, mapping[key], value)
    return inflated
