# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os

import torch


PIPPY_VERBOSITY = os.environ.get("PIPPY_VERBOSITY")
if PIPPY_VERBOSITY not in [None, "WARNING", "INFO", "DEBUG"]:
    logging.warning(f"Unsupported PIPPY_VERBOSITY level: {PIPPY_VERBOSITY}")
    PIPPY_VERBOSITY = None

if PIPPY_VERBOSITY:
    logging.getLogger("pippy").setLevel(PIPPY_VERBOSITY)
    # It seems we need to print something to make the level setting effective
    # for child loggers. Doing it here.
    logging.warning(f"Setting PiPPy logging level to: {PIPPY_VERBOSITY}")


def friendly_debug_info(v):
    if isinstance(v, torch.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad})"
    else:
        return str(v)


def map_debug_info(a):
    return torch.fx.node.map_aggregate(a, friendly_debug_info)
