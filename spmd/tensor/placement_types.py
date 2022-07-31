# Copyright (c) Meta Platforms, Inc. and affiliates
import torch.distributed.distributed_c10d as c10d

from dataclasses import dataclass
from typing import Optional, List, Sequence, cast
from spmd.tensor.device_mesh import DeviceMesh


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
    reduce_op: c10d.ReduceOp = c10d.ReduceOp.SUM


# used internally to propagate the placements
@dataclass
class PlacementSpec(object):
    __slots__ = ["ndim", "mesh", "placements"]
    ndim: int
    mesh: DeviceMesh
    placements: Sequence[Placement]

    @property
    def dims_map(self) -> List[int]:
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                r[shard_dim] = i
        return r

    @classmethod
    def from_dims_map(
        cls, mesh: DeviceMesh, dims_map: List[int], sums: List[int]
    ) -> "PlacementSpec":
        # by default replicate on device mesh dims
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

        for i, m in enumerate(dims_map):
            if m >= 0:
                if not placements[m].is_replicate():
                    raise RuntimeError(
                        "DeviceMesh cann't be mapped to two dimension of the same tensor"
                    )
                placements[m] = Shard(i)

        for s in sums:
            placements[s] = _Partial()

        spec = cls(len(dims_map), mesh, placements)
        return spec
