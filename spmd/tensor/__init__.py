# Copyright (c) Meta Platforms, Inc. and affiliates
#
# Keep all the DTensor imports in the spmd/tensor package,
# for the following reasons:
# 1. for BC purposes, existing works using spmd.tensor
#    imports still work as before
# 2. for compiler stack where we need to extend DTensor
#    functionality, this is our extension point.

from typing import Optional, Sequence, Callable, cast

import torch
import torch.nn as nn

# DTensor imports from core distributed
from torch.distributed._tensor import *
from torch.distributed._tensor.device_mesh import DeviceMesh, get_global_device_mesh
from torch.distributed._tensor.ops.utils import register_prop_rule
