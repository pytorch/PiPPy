# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os

import torch


PIPPY_VERBOSITY = os.environ.get("PIPPY_VERBOSITY", "OFF")

if PIPPY_VERBOSITY == "DEBUG":
    logging.getLogger("pippy").setLevel(logging.DEBUG)
elif PIPPY_VERBOSITY == "INFO":
    logging.getLogger("pippy").setLevel(logging.INFO)
elif PIPPY_VERBOSITY == "OFF":
    pass
else:
    print(f"[PiPPy] Unsupported PIPPY_VERBOSITY level: {PIPPY_VERBOSITY}")

print(f"[PiPPy] Setting logging level to: {PIPPY_VERBOSITY}")


def friendly_debug_info(v):
    if isinstance(v, torch.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad})"
    else:
        return str(v)


def map_debug_info(a):
    return torch.fx.node.map_aggregate(a, friendly_debug_info)
