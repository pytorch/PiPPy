import torch.distributed.distributed_c10d as c10d

from dataclasses import dataclass


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
