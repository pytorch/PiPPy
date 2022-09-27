# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from typing import Mapping, List
from torch.distributed._shard.checkpoint.metadata import (
    STATE_DICT_TYPE,
)
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from spmd import  DTensor as DT

def keep_visiting_tensors(value):
    return isinstance(value, torch.Tensor)

def traverse_state_dict(state_dict: STATE_DICT_TYPE, visitor, keep_traversing=keep_visiting_tensors):
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.

    Traversal is shortcuted when if finds a collection for which `keep_visiting_tensors` evaluates
    to false for all elements.

    By default, all collections with at least one ``torch.Tensor`` element are traversed.

    Visitor takes a path argument that is a tuple of the keys used to reach it.
    """
    # a value is terminal if it has no other containers values inside it
    def _is_terminal(value):
        values = None
        if isinstance(value, Mapping):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path, value):
        if _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)



def set_element(root_dict, path, value):
    cur_container = root_dict

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        if type(key) == str:
            cur_container = cur_container.setdefault(prev_key, {})
        else:
            cur_container = cur_container.setdefault(prev_key, [])

    key = path[-1]
    if type(key) == int:
        while len(cur_container) <= key:
            cur_container.append(None)
    cur_container[key] = value


def get_element(root_dict, path, default_value=None):
    cur_value = root_dict
    for part in path:
        if part not in cur_value:
            return default_value
        cur_value = cur_value[part]
    return cur_value


def _print_nested(value, padding="", prefix="", print_fun=print):
    if isinstance(value, ShardedTensor):
        print_fun(f"{padding}{prefix} ShardedTensor size {value.size()}")
        for shard in value.local_shards():
            _print_nested(shard.tensor, f"{padding}\t", f"{shard.metadata.shard_offsets} ", print_fun=print_fun)
    elif isinstance(value, DT):
        print_fun(f"{padding}{prefix} DistributedTensor size {value.size()}")
        # for shard in value.local_shards():
        _print_nested(value.local_tensor, f"{padding}\t", f"(offset ???) ", print_fun=print_fun)
    else:
        print_fun(f"{padding}{prefix} Tensor size {value.size()}")

def print_tensor(path, value, print_fun=print):
    _print_nested(path, value, print_fun=print_fun)


def element_wise_add(a: List[int], b: List[int]) -> List[int]:
    return [i_a + i_b for i_a, i_b in zip(a,b)]
