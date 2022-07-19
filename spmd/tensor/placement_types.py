# Copyright (c) Meta Platforms, Inc. and affiliates
import torch.distributed.distributed_c10d as c10d

from dataclasses import dataclass
from typing import Optional


@dataclass
class Placement(object):
    # base class Placement type
    pass


@dataclass
class Shard(Placement):
    # shard placement, shard on a dim
    dim: int


@dataclass
class Replicate(Placement):
    # replicate placement
    pass


@dataclass
class _Partial(Placement):
    # partial placement with reduce op
    reduce_op: c10d.ReduceOp


# convenient utils to check for placement types
def is_shard(placement: Placement, dim: Optional[int] = None) -> bool:
    is_shard = isinstance(placement, Shard)
    return is_shard and placement.dim == dim if dim is not None else is_shard

def is_replicate(placement: Placement) -> bool:
    return isinstance(placement, Replicate)

def is_partial(placement: Placement) -> bool:
    return isinstance(placement, _Partial)
