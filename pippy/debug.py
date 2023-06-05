# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import pippy.fx


def friendly_debug_info(v):
    if isinstance(v, torch.Tensor):
        return f"Tensor(size={v.shape})"
    else:
        return str(v)


def map_debug_info(a):
    return pippy.fx.node.map_aggregate(a, friendly_debug_info)
