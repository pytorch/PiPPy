# Copyright (c) Meta Platforms, Inc. and affiliates
import warnings
from typing import List, Optional, Iterable, Sequence
import torch
from torch.distributed.distributed_c10d import (
    get_rank,
    get_world_size,
    ReduceOp,
    _get_default_group,
    _reduce_scatter_base,
    new_group,
    ProcessGroup,
)

# autograd enabled collective
from torch.distributed.nn.functional import (
    all_gather,
    _all_gather_base,
    all_reduce,
    broadcast,
    scatter,
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
    DistributedTensor implementation.

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

    def __init__(
        self,
        device_type: str,
        mesh: Iterable[Sequence[int]],
        dim_groups: Optional[List[ProcessGroup]] = None,
    ) -> None:
        self.device_type = device_type
        self.mesh = torch.tensor(mesh, dtype=torch.int)
        default_pg = _get_default_group()
        backend_name = default_pg._get_backend_name()
        # TODO: if user want to pass pg_options, offer a way to do it
        # check default pg backend, should support device_type
        if device_type == "cpu":
            assert (
                backend_name == "gloo"
            ), f"ProcessGroup backend: {backend_name} not supporting CPU!"
        elif device_type == "cuda":
            if backend_name == "gloo":
                warnings.warn(
                    "We recommend using nccl backend for cuda device type, gloo backend might only have partial support!"
                )
            assert backend_name == "gloo" or backend_name == "nccl"
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

        # groups created by dimension, each dimension should have exact
        # one valid process group per rank
        self._dim_groups: List[ProcessGroup] = []
        if dim_groups is not None:
            # if user hand creating dimension based groups
            # we just take it and use it for communication
            # we assume user passing dim_gruops are legit
            # TODO: add more checks to check the correctness
            # of user passed in dim groups
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
                # local_subgroup_by_dim = self._dim_to_groups[dim]
                # swap the current dim to the last dim
                # then reshape to flatten out other dims
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(
                    -1, self.mesh.size(dim)
                )

                # multi-dim mesh, create subgroups by
                # looping over the transposed stacked_mesh
                # and append the groups
                for dim_mesh in pg_ranks_by_dim:
                    subgroup_ranks = dim_mesh.tolist()
                    # call new_group regardless of the current rank in the
                    # pg or not, it's required that all ranks participate
                    # in subgroup construction
                    new_subgroup = new_group(
                        ranks=subgroup_ranks,
                        backend=backend_name,
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
        return _get_default_group()._get_backend_name()

    def get_rank(self) -> int:
        return get_rank()

    def scatter(
        self, tensors: List[torch.Tensor], src: int = 0
    ) -> torch.Tensor:
        return scatter(tensors, src=src)

    # pyre-fixme[3]: Return type must be annotated.
    def broadcast(self, tensor: torch.Tensor, mesh_dim: int = 0):
        return broadcast(tensor, src=0, group=self._dim_groups[mesh_dim])

    # pyre-fixme[3]: Return type must be annotated.
    def all_gather(self, tensor: torch.Tensor):
        return all_gather(tensor)

    # pyre-fixme[3]: Return type must be annotated.
    def all_gather_base(
        self, output_tensor: torch.Tensor, tensor: torch.Tensor
    ):
        # only nccl have all_gather base
        if self.backend() == "nccl":
            return _all_gather_base(output_tensor, tensor)
        else:
            # if not nccl, fallback to use all_gather
            # and reform the output tensor
            gathered_chunks = self.all_gather(tensor)
            # TODO: find more performant way
            for chunk in gathered_chunks:
                output_tensor.copy_(torch.cat(gathered_chunks))
            return output_tensor

    # pyre-fixme[3]: Return type must be annotated.
    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM):
        return all_reduce(tensor, op=op)

    # pyre-fixme[3]: Return type must be annotated.
    def reduce_scatter_base(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ):
        # NOTE: two caveats:
        # 1. only NCCL support reduce_scatter
        # 2. we are using a non-autograd enabled reduce_scatter
        #    this is fine as we are using it now on redistribute
        #    which have backward enabled, but if we want to use
        #    this in other case which requires autograd, we should
        #    add the autograd enabled collective in distributed/nn/functional
        if self.backend() == "nccl":
            _reduce_scatter_base(output, input, op)
            return output
        else:
            # it's gloo, which does not have reduce_scatter
            # we have to do all_reduce + scatter
            reduced_tensor = self.all_reduce(input)
            shard_dim = 0
            num_chunks = 0
            for i in range(input.ndim):
                if input.size(i) > output.size(i):
                    num_chunks = input.size(i) // output.size(i)
                    shard_dim = i
                    break

            chunks = reduced_tensor.chunk(num_chunks, dim=shard_dim)
            return self.scatter(chunks)
