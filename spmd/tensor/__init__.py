# Copyright (c) Meta Platforms, Inc. and affiliates
#
# Keep all the DTensor imports in the spmd/tensor package,
# for the following reasons:
# 1. for BC purposes, existing works using spmd.tensor
#    imports still work as before
# 2. for compiler stack where we need to extend DTensor
#    functionality, this is our extension point.

# DTensor imports from core distributed
from torch.distributed._tensor import *  # noqa: F403
from torch.distributed._tensor.device_mesh import (
    DeviceMesh,
    get_global_device_mesh,
)
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.dispatch import (
    _CURRENT_DECOMPOSITION_TABLE,
    operator_dispatch,
)
from torch.distributed._tensor.placement_types import Placement, _Partial
from torch.distributed._tensor.redistribute import (
    _redistribute_with_local_tensor,
)

# experimental ops import
from .experimental_ops import *  # noqa: F403
