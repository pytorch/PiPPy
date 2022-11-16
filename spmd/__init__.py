# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor import *  # noqa: F401, F403
from spmd.tensor.device_mesh import DeviceMesh
from spmd.compiler.api import Schema, SPMD
from spmd.tensor.placement_types import Shard, Replicate


# All public APIs from spmd package
__all__ = [
    "DeviceMesh",
    "Replicate",
    "Schema",
    "SPMD",
    "Shard",
    "distribute_module",
]
