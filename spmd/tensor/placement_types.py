# Copyright (c) Meta Platforms, Inc. and affiliates
import torch.distributed.distributed_c10d as c10d

from dataclasses import dataclass
from typing import Optional


@dataclass
class Placement(object):
    # base class Placement type

    # convenient utils to check for placement types
    def is_shard(self, dim: Optional[int] = None) -> bool:
        if dim is not None and isinstance(self, Shard):
            return self.dim == dim
        else:
            return isinstance(self, Shard)

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, _Partial)


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
