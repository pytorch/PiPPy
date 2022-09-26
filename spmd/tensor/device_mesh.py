# Copyright (c) Meta Platforms, Inc. and affiliates
import warnings
from typing import List, Optional, Iterable, Sequence, Union, TypeAlias, TypeVar
import torch
import torch.nn.functional as F
from torch.distributed._spmd.comm_tensor import CommTensor
from torch.distributed.distributed_c10d import (
    all_gather,
    all_reduce,
    broadcast,
    get_rank,
    get_world_size,
    get_global_rank,
    ReduceOp,
    GroupMember,
    scatter,
    _get_default_group,
    reduce_scatter,
    new_group,
    ProcessGroup,
)

_global_device_mesh: Optional["DeviceMesh"] = None


def get_global_device_mesh() -> "DeviceMesh":
    global _global_device_mesh
    assert (
        _global_device_mesh is not None
    ), "Could not get a default device mesh!"
    return _global_device_mesh


def set_global_device_mesh(mesh: Optional["DeviceMesh"]) -> None:
    global _global_device_mesh
    _global_device_mesh = mesh


T = TypeVar('T')
_L: TypeAlias = Union[T, Sequence[T]]
DeepList = _L[_L[_L[_L[_L[int]]]]]

MeshExprT: TypeAlias = Union[
    torch.Tensor,
    DeepList,
]


class DeviceMesh(object):
    """
    DeviceMesh represents a mesh of devices, where layout of devices could be
    represented as a n-d dimension array, and each value of the n-d dimensional
    array is the global id of the default process group ranks.

    DeviceMesh could be used to describe the layout of devices across the cluster,
    and serves as a proxy for communication among the device lists within the cluster.

    We use the default ProcessGroup in this DeviceMesh class to implement proper
    communications. Note that we also add collective wrappers in this class. This is
    used to decouple detailed communication backend with the underlying
    DTensor implementation.

    DeviceMesh can be used as a context manager.
    Args:
        device_type (str): device type of the mesh. Currently supports: cpu, cuda.
        mesh (ndarray): could be a multi-dimension array or an integer tensor that
            describes the layout of devices, the ids are global ids of the
            default process group.
        dim_groups (List[ProcessGroup], optional): The ProcessGroup used per mesh
            dimension.

    Returns:
        A :class:`DeviceMesh` object

    Example (2 host with 4 GPUs each):
        ```
        # The following program runs on each process/rank in SPMD manner.
        # initialized default world
        torch.distributed.init_process_group(backend="nccl", world_size=8)
        # initialize device mesh as (2, 4) to represent the topology
        # of cross-host(dim 0), and within-host (dim 1)
        mesh = DeviceMesh(device_type="cuda",
                          mesh=[
                            [0, 1, 2, 3],
                            [4, 5, 6, 7]
                          ])
        ```
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7)

    """

    device_type: str
    mesh: torch.Tensor
    _backend: str

    @classmethod
    def _copy_mesh(cls, mesh: MeshExprT) -> torch.Tensor:
        return torch.as_tensor(mesh, dtype=torch.int).clone().detach()

    def __init__(
        self,
        device_type: str,
        mesh: MeshExprT,
        dim_groups: Optional[List[ProcessGroup]] = None,
    ) -> None:
        self.device_type = device_type
        self.mesh = self._copy_mesh(mesh)
        default_pg = _get_default_group()
        self._backend = default_pg._get_backend_name()
        # TODO: if user want to pass pg_options, offer a way to do it
        # check default pg backend, should support device_type
        if device_type == "cpu":
            assert (
                self._backend == "gloo"
            ), f"ProcessGroup backend: {self._backend} not supporting CPU!"
        elif device_type == "cuda":
            if self._backend == "gloo":
                warnings.warn(
                    "We recommend using nccl backend for cuda device type, gloo backend might only have partial support!"
                )
            assert self._backend == "gloo" or self._backend == "nccl"
        else:
            raise RuntimeError(
                f"DeviceMesh only support cpu or cuda device type, but got {device_type}"
            )

        world_size = get_world_size()
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!"
            )

        unique_mesh_values = self.mesh.unique(sorted=True)
        if unique_mesh_values.numel() != self.mesh.numel():
            raise RuntimeError(
                f"DeviceMesh cannot have duplicate values, but found {self.mesh.tolist()}"
            )

        # coordinates of this rank on the mesh
        rank_coords = (self.mesh == get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinate_on_dim: Optional[List[int]] = (
            rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
        )

        # groups created by dimension, each dimension should have exact
        # one valid process group per rank
        self._dim_groups: List[ProcessGroup] = []
        if dim_groups is not None:
            # if user hand creating dimension based groups
            # we just take it and use it for communication
            if not isinstance(dim_groups, list):
                raise RuntimeError(
                    "dim_groups expected to be Optional[List[ProcessGroup]]"
                )

            for group in dim_groups:
                if not isinstance(group, ProcessGroup):
                    raise RuntimeError(
                        f"found object in dim_groups that is not a ProcessGroup: {group}"
                    )

            if self.get_rank() in self.mesh:
                if len(dim_groups) != self.mesh.ndim:
                    raise RuntimeError(
                        f"length of dim_groups ({len(dim_groups)}) expected to be equal to mesh.ndim ({self.mesh.ndim})"
                    )
            else:
                if len(dim_groups) != 0:
                    raise RuntimeError(
                        f"length of dim_groups ({len(dim_groups)}) expected to be equal to 0 on rank {self.get_rank()} for mesh {self.mesh}"
                    )

            self._dim_groups = dim_groups
            return

        if self.mesh.ndim == 1 and unique_mesh_values[-1] == world_size - 1:
            # if the mesh is the same as world_pg, we just append the default
            # pg to the first dim goups, as new_group cannot have the exact
            # same ranks as world
            self._dim_groups.append(default_pg)
        else:
            # create sub pgs base on the mesh argument specified
            # handle multi-dim mesh, create subgroups by
            # looping over the pg_ranks_by_dim for each dim
            for dim in range(self.mesh.ndim):
                # swap the current dim to the last dim
                # then reshape to flatten out other dims
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(
                    -1, self.mesh.size(dim)
                )

                # multi-dim mesh, create subgroups by
                # looping over the pg_ranks for each dim
                # and append the groups
                for dim_mesh in pg_ranks_by_dim:
                    subgroup_ranks = dim_mesh.tolist()
                    # call new_group regardless of the current rank in the
                    # pg or not, it's required that all ranks participate
                    # in subgroup construction
                    new_subgroup = new_group(
                        ranks=subgroup_ranks, backend=self._backend
                    )
                    # only add to dim_groups if the current rank in the subgroup
                    if self.get_rank() in subgroup_ranks:
                        if len(self._dim_groups) > dim:
                            raise RuntimeError(
                                f"Each device mesh dimension should get only one process group, but got {self.get_rank} in {subgroup_ranks}!"
                            )
                        self._dim_groups.append(new_subgroup)

    def __enter__(self) -> "DeviceMesh":
        # set global device_mesh to this instance
        set_global_device_mesh(self)
        return self

    # pyre-fixme[2]: Parameter must be annotated.
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # unset global device mesh
        set_global_device_mesh(None)

    def __repr__(self) -> str:
        return f"DeviceMesh:({self.mesh.tolist()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceMesh):
            return False
        if id(self) == id(other):
            return True
        return self.mesh.equal(other.mesh)

    def get_dim_groups(self) -> List[ProcessGroup]:
        return self._dim_groups

    # pyre-fixme[3]: Return type must be annotated.
    def size(self, dim: int = 0):
        return self.mesh.size(dim)

    @property
    def ndim(self) -> int:
        return self.mesh.ndim

    def backend(self) -> str:
        return self._backend

    def get_rank(self) -> int:
        return get_rank()

    def get_coordinate_on_dim(self, dim: int) -> Optional[int]:
        """
        Return the relative index of this rank relative to a given
        dimension of the mesh. If this rank is not part of the mesh, return None.
        """
        return self._coordinate_on_dim[dim] if self._coordinate_on_dim else None

    def _pad_tensor_dim_by_1(
        self, tensor: torch.Tensor, dim: int
    ) -> torch.Tensor:
        pad = [0, 0] * (tensor.ndim - dim)
        pad[-1] = 1
        return F.pad(tensor, pad)

    def _unpad_tensor_dim_by_1(
        self, tensor: torch.Tensor, dim: int
    ) -> torch.Tensor:
        return tensor.narrow(dim, start=0, length=tensor.size(dim) - 1)

    def scatter(
        self,
        tensor_to_scatter: torch.Tensor,
        mesh_dim: int = 0,
        tensor_dim: int = 0,
    ) -> torch.Tensor:
        """
        scatter a big tensor to a device mesh dimension. We by default
        use the first rank of the mesh dimension as the source of truth, i.e
        for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
        scatter the tensor splitted on rank 0 to rank 0/1, and tensor splitted
        on rank 2 to rank 2/3.

        Args:
            tensor_to_scatter (torch.Tensor): the tensor to be scattered.
            mesh_dim (int, optional): indicate which mesh dimension we want
                to scatter on, we by default choose the first rank on the
                mesh dimension as source of truth.
            tensor_dim (int, optional): indicate which tensor dimension we want
                to split the `tensor_to_scatter` before scattering.

        Returns:
            A :class:`torch.Tensor` object
        """
        my_coordinate = self.get_coordinate_on_dim(mesh_dim)
        # TODO: what should happen if rank is not in the mesh?
        assert (
            my_coordinate is not None
        ), "Rank if not part of mesh"  # TODO: figure out behavior here

        num_chunks = self.size(mesh_dim)
        dim_group = self._dim_groups[mesh_dim]
        # src need to be global rank
        src_for_dim = 0
        if dim_group is not GroupMember.WORLD:
            src_for_dim = get_global_rank(dim_group, 0)

        scatter_list = list(
            tensor_to_scatter.tensor_split(num_chunks, dim=tensor_dim)
        )
        # gloo does not support uneven scattering, nccl supports it, but it
        # might be slow compared to even size collective, we need to pad tensor
        # before really calling scatter, and unpad/narrow it after the collective
        # TODO: consider if we should remove this logic once ProcessGroupGloo
        # suupport uneven list, and collective perfomance on par
        idx_start_to_pad = tensor_to_scatter.size(tensor_dim) % num_chunks
        to_scatter = []
        for i, scatter_tensor in enumerate(scatter_list):
            if idx_start_to_pad != 0 and i >= idx_start_to_pad:
                scatter_tensor = self._pad_tensor_dim_by_1(
                    scatter_tensor, tensor_dim
                )
            scatter_tensor = scatter_tensor.contiguous()
            to_scatter.append(CommTensor(scatter_tensor))

        # N.B.: the `tensor` below will be a CommTensor too due to CommTensor's
        # propagation rule: propagte wrapping until communication is called.
        # This is necessary in order to properly trigger CommTensor's dispatch
        # function for scatter_.
        tensor = torch.empty_like(to_scatter[my_coordinate])
        if src_for_dim == get_rank():
            scatter(
                tensor,
                scatter_list=to_scatter,
                src=src_for_dim,
                group=dim_group,
            )
        else:
            scatter(tensor, scatter_list=None, src=src_for_dim, group=dim_group)

        # resize to uneven size if needed
        if idx_start_to_pad != 0 and my_coordinate >= idx_start_to_pad:
            # tensor = tensor.narrow(tensor_dim, start=0, length=tensor.size(tensor_dim) - 1)
            tensor = self._unpad_tensor_dim_by_1(tensor, tensor_dim)
        return tensor

    def broadcast(
        self, tensor: torch.Tensor, mesh_dim: int = 0
    ) -> torch.Tensor:
        """
        broadcast the tensor to a device mesh dimension. We by default
        use the first rank of the mesh dimension as the source of truth, i.e
        for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will
        broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2
        to rank 2/3.

        Args:
            tensor (torch.Tensor): tensor to scatter.
            mesh_dim (int, optional): indicate which mesh dimension we want
                to scatter on, we by default choose the first rank on the
                mesh dimension as source of truth.

        Returns:
            A :class:`torch.Tensor` object
        """
        dim_group = self._dim_groups[mesh_dim]
        # src need to be global rank
        src_for_dim = 0
        if dim_group is not GroupMember.WORLD:
            src_for_dim = get_global_rank(dim_group, 0)

        # CommTensor does not change eager mode behavior. During tracing, it
        # makes sure communication result is properly waited before subsequent
        # read operations.
        if not tensor.is_contiguous():
            tensor = CommTensor(tensor.contiguous())
        else:
            tensor = CommTensor(tensor.clone())
        broadcast(tensor, src=src_for_dim, group=dim_group)
        return tensor

    def all_gather(
        self,
        tensor: torch.Tensor,
        output_shape: Sequence[int],
        mesh_dim: int = 0,
        tensor_dim: int = 0,
    ) -> torch.Tensor:
        """
        all_gather the tensor on each rank to a bigger output_tensor on a
        device mesh dimension.

        Args:
            tensor (torch.Tensor): tensor to be gathered on each rank.
            output_shape (Tuple[int, ...]): output shape of the all_gather call,
                this is needed because we need to know the proper size of each
                receving tensor on each rank in order to call dist.all_gather,
                as sometimes we have uneven sharding on the mesh dimension. We
                use torch.tensor_split semantics to construct the recv tensor
                and cat them to the output_shape.
            mesh_dim (int, optional): indicate which mesh dimension we want
                to scatter on, we by default choose the first rank on the
                mesh dimension as source of truth.
            tensor_dim (int, optional): indicate which tensor dimension we want
                to concat on after we gather list of tensors from all rank.

        Returns:
            A :class:`torch.Tensor` object
        """
        num_chunks = self.size(mesh_dim)
        _, rem = divmod(output_shape[tensor_dim], num_chunks)
        assert rem == 0, "output_shape must be divisible by num_chunks"
        input_tensor = tensor.contiguous()
        gathered_list = [
            CommTensor(torch.empty_like(input_tensor))
            for _ in range(num_chunks)
        ]

        dim_group = self._dim_groups[mesh_dim]
        # N.B. CommTensor does not change eager mode behavior. During tracing, it
        # makes sure communication result is properly waited before subsequent
        # read operations.
        # input tensor must be contiguous
        all_gather(
            gathered_list,
            CommTensor(input_tensor),
            group=dim_group,
        )
        return torch.cat(gathered_list, dim=tensor_dim)  # type: ignore

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,  # type: ignore
        mesh_dim: int = 0,
    ) -> torch.Tensor:
        """
        all_reduce the tensor on each rank on a device mesh dimension, and
        return an output tensor on each rank after all_reduce.

        Args:
            input (torch.Tensor): tensor to be all_reduced on each rank.
            op (:class:`torch.distributed.distributed_c10d.ReduceOp, optional):
                the reduction op of all_reduce (i.e. ReduceOp.SUM)
            mesh_dim (int, optional): indicate which mesh dimension we want
                to reduce on.

        Returns:
            A :class:`torch.Tensor` object
        """
        dim_group = self._dim_groups[mesh_dim]
        # CommTensor does not change eager mode behavior. During tracing, it
        # makes sure communication result is properly waited before subsequent
        # read operations.
        if not tensor.is_contiguous():
            tensor = CommTensor(tensor.contiguous())
        else:
            tensor = CommTensor(tensor.clone())
        all_reduce(tensor, op=op, group=dim_group)
        return tensor

    def reduce_scatter(
        self,
        input: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,  # type: ignore
        mesh_dim: int = 0,
        tensor_dim: int = 0,
    ) -> torch.Tensor:
        """
        reduce_scattter the tensor on each rank on a device mesh dimension, and
        return an output tensor that's scattered to each rank after reduce.

        Args:
            input (torch.Tensor): tensor to be reduced and scattered on each rank.
            op (:class:`torch.distributed.distributed_c10d.ReduceOp, optional):
                the reduction op of reduce_scatter (i.e. ReduceOp.SUM)
            mesh_dim (int, optional): indicate which mesh dimension we want
                to scatter on.
            tensor_dim (int, optional): indicate which tensor dimension we want
                to scatter after the reduction.

        Returns:
            A :class:`torch.Tensor` object
        """
        my_coordinate = self.get_coordinate_on_dim(mesh_dim)
        # TODO: what should happen if rank is not in the mesh?
        assert (
            my_coordinate is not None
        ), "Rank if not part of mesh"  # TODO: figure out behavior here

        num_chunks = self.size(mesh_dim)
        if self._backend == "nccl":
            input_list = list(input.tensor_split(num_chunks, dim=tensor_dim))
            # gloo does not support uneven scattering, nccl supports it, but it
            # might be slow compared to even size collective, we need to pad tensor
            # before really calling scatter, and unpad/narrow it after the collective
            # TODO: consider if we should remove this logic once ProcessGroupGloo
            # suupport uneven list, and collective perfomance on par
            idx_start_to_pad = input.size(tensor_dim) % num_chunks
            to_scatter = []
            for i, scatter_tensor in enumerate(input_list):
                if idx_start_to_pad != 0 and i >= idx_start_to_pad:
                    scatter_tensor = self._pad_tensor_dim_by_1(
                        scatter_tensor, tensor_dim
                    )
                scatter_tensor = scatter_tensor.contiguous()
                to_scatter.append(CommTensor(scatter_tensor))

            output = torch.empty_like(to_scatter[my_coordinate])
            dim_group = self._dim_groups[mesh_dim]
            reduce_scatter(output, to_scatter, op=op, group=dim_group)

            # resize to uneven size if needed
            if idx_start_to_pad != 0 and my_coordinate >= idx_start_to_pad:
                # tensor = tensor.narrow(tensor_dim, start=0, length=tensor.size(tensor_dim) - 1)
                output = self._unpad_tensor_dim_by_1(output, tensor_dim)

            return output
        elif self._backend == "gloo":
            # it's gloo, which does not have reduce_scatter
            # we have to do all_reduce + scatter
            warnings.warn(
                "ProcessGroupGloo does not support reduce_scatter, falling back with all reduce!"
            )
            reduced_tensor = self.all_reduce(input, op=op, mesh_dim=mesh_dim)
            chunks = reduced_tensor.tensor_split(num_chunks, dim=tensor_dim)
            return chunks[my_coordinate]
        else:
            raise RuntimeError(
                f"backend {self._backend} does not support reduce_scatter!"
            )
