# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import torch
from torch.utils._pytree import tree_map
from typing import Dict, List, Callable, Optional
from spmd.tensor.device_mesh import get_global_device_mesh, DeviceMesh
from spmd.tensor.placement_types import Placement, Shard, Replicate, _Partial


class Tensor(torch.Tensor):
    __slots__ = ["_local_tensor", "_device_mesh", "_placements"]

    # class attribute that handles ops, all handled
    # ops should appear in this table
    _dist_tensor_dispatch_ops: Dict[str, Callable] = {}

    # context = contextlib.nullcontext

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        device_mesh: DeviceMesh,
        placements: List[Placement],
        size: torch.Size,
        **kwargs,
    ):
        # new method instruct wrapper tensor and add placement spec
        # it does not do actual distribution, __init__ should do it instead.
        # TODO: implement __init__ for tensor constructors
        assert isinstance(placements, list)
        assert len(placements) == 1, "Only support 1-d placement for now"
        # sizes = _flatten_tensor_size(size)
        dtype = kwargs["dtype"]
        layout = kwargs["layout"]
        requires_grad = kwargs["requires_grad"]
        device = kwargs.get("device", "cpu")

        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            size,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=requires_grad,
        )
        r._device_mesh = device_mesh
        # deepcopy and set spec, data should be handled
        # by __init__ or from_local instead.
        r._placements = copy.deepcopy(placements)
        return r

    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"DistributedTensor({self._local_tensor}, placements={self._placements})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap_mesh(e):
            # if this tensor is not Distributed, then return none. We will reinterpret it as replicated
            if not isinstance(e, Tensor):
                return None
            return e.device_mesh

        def unwrap(e):
            return e._local_tensor if isinstance(e, Tensor) else e

        def wrap(e, mesh, placements):
            return (
                Tensor.from_local(e, mesh, placements, run_check=False)
                if isinstance(e, torch.Tensor)
                else e
            )

        args_mesh = tree_map(unwrap_mesh, args)
        # assert all_equal(spec.device_mesh for spec in args_spec), "can't compuate across different meshes"
        # for spec in args_spec:
        #     assert spec.device_mesh.mesh.ndim == 1, "Only 1-D mesh supported now"

        # take a short cut if all arguments are replicated
        all_replicated = True
        for arg in args:
            if isinstance(arg, Tensor):
                all_replicated &= arg.placements[0] == Replicate()

        if all_replicated:
            rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            return wrap(rs, args_mesh[0], args[0].placements)

        if str(func) in Tensor._dist_tensor_dispatch_ops:
            # dispatch to distributed tensor ops
            return Tensor._dist_tensor_dispatch_ops[str(func)](*args, **kwargs)
        else:
            # default to local tensor ops, this is wrong
            # but we use it now to enable more tensor property access
            rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            rs = wrap(rs, args_mesh[0], args[0].placements)
            return rs

    @classmethod
    def from_local(
        cls, local_tensor, device_mesh=None, placements=None, run_check=True
    ):
        # if same shape/dtype, no need to run_check, if not, must allgather
        # the metadatas to check the size/dtype across ranks
        # There should be no data communication unless there's replication
        # strategy, where we broadcast the replication from rank 0
        device_mesh = (
            get_global_device_mesh() if device_mesh is None else device_mesh
        )
        # convert the local tensor to desired device base on device mesh's device_type
        local_tensor = local_tensor.to(device_mesh.device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in device_mesh.ndim]

        tensor_shape = list(local_tensor.size())
        for idx, placement in enumerate(placements):
            # device_list = device_mesh.mesh[idx]
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                # recover tensor shape on the shard dim
                tensor_shape[shard_dim] = tensor_shape[
                    shard_dim
                ] * device_mesh.size(idx)
            elif isinstance(placement, Replicate):
                # broadcast rank 0 tensor to all ranks
                local_tensor = local_tensor.contiguous()
                device_mesh.broadcast(local_tensor, 0)
            elif isinstance(placement, _Partial):
                # we don't need to do anything to Partial case
                pass
            else:
                raise RuntimeError(
                    f"placement type {type(placement)} not supported!"
                )

        if run_check:
            # TODO: by default check tensor metas across rank
            pass

        dist_tensor = cls(
            device_mesh,
            placements,
            torch.Size(tensor_shape),
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=local_tensor.requires_grad,
        )
        dist_tensor._local_tensor = local_tensor
        return dist_tensor

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[List[Placement]] = None,
    ) -> "Tensor":
        # This API perform necessary transformations and get
        # a new DistributedTensor with the new spec. i.e. for
        # sharding it's a reshard behavior.
        # TODO: handle last shard uneven with padding
        # right now we assume all local shard equal size
        device_mesh = (
            get_global_device_mesh() if device_mesh is None else device_mesh
        )
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        current_placements = self.placements
        assert len(placements) == 1, "Only support 1-d placement for now"
        assert self.device_mesh.mesh.equal(
            device_mesh.mesh
        ), "cross mesh comm not support yet"
        local_tensor = self.local_tensor()
        if isinstance(current_placements[0], Shard) and isinstance(
            placements[0], Replicate
        ):
            # for shard, all_gather all shards and return the global tensor
            global_tensor = torch.empty(
                *self.size(), device=local_tensor.device, dtype=self.dtype
            )
            # NOTE: all_gather_base only works well when tensor
            # sharded on a sequential list of devices
            device_mesh.all_gather_base(global_tensor, local_tensor)
            replica_tensor = Tensor.from_local(
                global_tensor, device_mesh, placements
            )
            replica_tensor._placements[0] = Replicate()
            return replica_tensor
        elif isinstance(current_placements[0], _Partial) and isinstance(
            placements[0], Replicate
        ):
            reduced_tensor = device_mesh.all_reduce(
                local_tensor, current_placements[0].reduce_op
            )
            replica_tensor = Tensor.from_local(
                reduced_tensor, device_mesh, current_placements
            )
            # change placement to replicate
            replica_tensor._placements[0] = Replicate()
            return replica_tensor
        elif current_placements == placements:
            return self
        else:
            raise RuntimeError(
                f"Converting from {current_placements} to {placements} not supported!"
            )

    def local_tensor(self) -> torch.Tensor:
        return self._local_tensor  # type: ignore

    @property
    def placements(self):
        # placement should be a read only propety
        # to disallow caller modification on it
        # caller who want a different PlacementSpec
        # should call redistribute instead.
        return self._placements

    @property
    def device_mesh(self):
        # device_mesh should be a read only propety
        # to disallow caller modification on it
        # caller who want a different device_mesh
        # should call redistribute instead.
        return self._device_mesh
