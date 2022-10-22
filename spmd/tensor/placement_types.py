# Copyright (c) Meta Platforms, Inc. and affiliates
from abc import ABC, abstractmethod

import torch
import torch.distributed.distributed_c10d as c10d

from dataclasses import dataclass
from typing import Optional, List, Sequence, Union, Tuple, cast
from spmd.tensor.device_mesh import DeviceMesh


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


class _ReduceOp(ABC):
    # ReduceOp that supports non elementwise reduction

    # a extension of c10d.ReduceOp to support custom reduction
    @abstractmethod
    def reduce_tensor(self, tensor: torch.Tensor, mesh, mesh_dim) -> torch.Tensor:
        # reduce tensor to a single tensor, by default elementwise reduction
        # is allowed
        ...

class _ElementWiseReduceOp(_ReduceOp):
    # ReduceOp that supports elementwise reduction
    

    # a extension of c10d.ReduceOp to support custom reduction
    def reduce_tensor(self, tensor: torch.Tensor, mesh, mesh_dim) -> torch.Tensor:
        # reduce tensor to a single tensor, by default elementwise reduction
        # is allowed
        ...


@dataclass
class _Partial(Placement):
    # partial placement with reduce op
    reduce_op: c10d.ReduceOp = c10d.ReduceOp.SUM  # type: ignore


    def to_replicate(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        mesh.all_reduce(
            tensor, self.reduce_op, mesh_dim=mesh_dim
        )
        return tensor


    def to_shard(self, tensor: torch.Tensor, shard_spec: Placement, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        assert shard_spec.is_shard()
        mesh.reduce_scatter(
            tensor, self.reduce_op, mesh_dim=mesh_dim
        )
        return tensor


# used internally to propagate the placements
@dataclass
class DTensorSpec(object):
    mesh: DeviceMesh
    placements: Sequence[Placement]
    # shape of the current dist tensor, this will be set upon
    # construction of the DTensor, prop rule could read it, and
    # would need to set in output spec when calculate the output
    # sharding
    shape: torch.Size
    # ndim of the current dist tensor, if passed in, this would be
    # validated with shape, if not passed in, will be generated from
    # the shape
    ndim: int = -1

    def __post_init__(self) -> None:
        if self.ndim == -1:
            self.ndim = len(self.shape)

    @property
    def dim_map(self) -> List[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 1, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `_Partial`, we have to
        explicitly deal with it, so that when we create a DTensorSpec
        with dim_map, we could properly record the pending sums.
        """
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                if r[shard_dim] > -1:
                    raise ValueError(
                        f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                        " DTensor operator implementation does not support things like hybrid"
                        " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
                    )
                r[shard_dim] = i
        return r

    @property
    def sums(self) -> List[int]:
        """
        sums is a property we derive from `placements` of the
        distributed tensor. It simply return a list of ints where
        sums[i] denotes the pending sum (partial) on mesh dim i
        """
        return [
            idx
            for idx, placement in enumerate(self.placements)
            if placement.is_partial()
        ]

    @property
    def local_shape(self) -> Tuple[int, ...]:
        """
        Compute the shape of a local shard of the given DTensor on its current
        global rank.
        """
        # TODO: support uneven sharding
        assert (
            self.shape is not None
        ), "DTensorSpec does not contain global shape."
        local_shape = list(self.shape)  # start with global shape
        for idx, placement in enumerate(self.placements):
            if isinstance(placement, Shard):
                assert (
                    local_shape[placement.dim] % self.mesh.size(idx) == 0
                ), f"Only even sharding supported for now. (Got {local_shape[placement.dim]} // {self.mesh.size(idx)} for mesh idx {idx}"
                local_shape[placement.dim] //= self.mesh.size(idx)
        return tuple(local_shape)

    @property
    def local_offsets(self) -> Tuple[int, ...]:
        """
        Compute the offsets of a local shard of the given DTensor on its current
        global rank. This is mostly used by distributed checkpointing to know the
        exact offsets of the local shard.
        """
        assert (
            self.shape is not None
        ), "DTensorSpec does not contain global shape."
        local_offsets = [0] * self.ndim
        for idx, mesh_dim in enumerate(self.dim_map):
            if mesh_dim > -1:
                my_coordinate = self.mesh.get_coordinate_on_dim(mesh_dim)
                # TODO: what should happen if rank is not in the mesh?
                # see issue https://github.com/pytorch/tau/pull/492
                assert (
                    my_coordinate is not None
                ), "Rank if not part of mesh"  # TODO: figure out behavior here
                mesh_dim_size = self.mesh.size(mesh_dim)
                quot, rem = divmod(self.shape[idx], mesh_dim_size)
                local_offsets[idx] = my_coordinate * quot + (
                    rem if my_coordinate >= rem else my_coordinate
                )

        return tuple(local_offsets)

    @classmethod
    def from_dim_map(
        cls,
        mesh: DeviceMesh,
        dim_map: List[int],
        sums: List[int],
        shape: torch.Size,
    ) -> "DTensorSpec":
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            shape (torch.Size): shape of the DTensor associated with this spec.

        Return:
            a class:`DTensorSpec` object
        """
        # by default replicate on device mesh dims
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # find all mesh dims that need pending reductions
        for s in sums:
            placements[s] = _Partial()

        for i, m in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    raise RuntimeError(
                        f"DeviceMesh dimension cann't be mapped to two dimension of the same tensor: {i} and {placement.dim}"
                    )
                elif placement.is_partial():
                    raise RuntimeError(
                        f"DeviceMesh dimension {m} cannot be both shard and partial!"
                    )
                placements[m] = Shard(i)

        return cls(mesh, placements, shape=shape, ndim=len(dim_map))


# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type sould
# be the same set of possiblities.
OutputSpecType = Optional[
    Union[DTensorSpec, Tuple[DTensorSpec, ...], List[DTensorSpec]]
]
