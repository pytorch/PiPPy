# Copyright (c) Meta Platforms, Inc. and affiliates
from spmd.tensor.api import DTensor
from spmd.tensor.device_mesh import DeviceMesh
from spmd.tensor.placement_types import Placement, Shard, Replicate, _Partial

# Import all builtin dist tensor ops
import spmd.tensor.ops
