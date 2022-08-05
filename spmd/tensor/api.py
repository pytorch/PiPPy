# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import torch
from torch.utils._pytree import tree_map, tree_flatten
from typing import Dict, Callable, Optional, Sequence
from spmd.tensor.device_mesh import get_global_device_mesh, DeviceMesh
from spmd.tensor.placement_types import Placement, Shard, Replicate, _Partial
from spmd.tensor.redistribute import Redistribute

"""
If set to true, __DEBUG_STRICT will fail when an op doesn't have a sharding rule registered.
"""
_DEBUG_STRICT = False


class Tensor(torch.Tensor):  # pyre-ignore[13]: pyre is bad at __new__
    _local_tensor: torch.Tensor
    _device_mesh: DeviceMesh
    _placements: Sequence[Placement]
    __slots__ = ["_local_tensor", "_device_mesh", "_placements"]

    # class attribute that handles ops, all handled
    # ops should appear in this table
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    _dist_tensor_dispatch_ops: Dict[str, Callable] = {}

    @staticmethod
    def __new__(
        cls,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        size: torch.Size,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> "Tensor":
        # new method instruct wrapper tensor and add placement spec
        # it does not do actual distribution, __init__ should do it instead.
        # TODO: implement __init__ for tensor constructors
        assert isinstance(placements, list)
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

    # pyre-fixme[14]: `__repr__` overrides method defined in `Tensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"DistributedTensor({self._local_tensor}, placements={self._placements})"

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # if we find nn.functional name in dispatch op, dispatch to it instead,
        # this allow us to override some python level behaviors that wouldn't be
        # possible in __torch_dispatch__ level.
        if str(func) in Tensor._dist_tensor_dispatch_ops:
            # dispatch to the same table as the name should be different between
            # torch_function and torch_dispatch
            return Tensor._dist_tensor_dispatch_ops[str(func)](*args, **kwargs)
        else:
            # if not, just do nothing here
            return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        # check that we are not getting mixed vanilla and Distributed tensors
        arg_list, arg_spec = tree_flatten(args)
        for arg in arg_list:
            if isinstance(arg, torch.Tensor) and not isinstance(arg, Tensor):
                raise RuntimeError(
                    f"{func}: got mixed distributed and non-distributed tensors."
                )

        def unwrap_mesh(e: Tensor) -> DeviceMesh:
            # if this tensor is not Distributed, then return none. We will reinterpret it as replicated
            if not isinstance(e, Tensor):
                return None
            mesh = e.device_mesh
            assert (
                mesh.ndim == 1
            ), "DistributedTensor ops not supporting multi-dim mesh yet"
            return mesh

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def unwrap(e):
            return e._local_tensor if isinstance(e, Tensor) else e

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
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
            if _DEBUG_STRICT:
                raise RuntimeError(
                    f"Operator {func} does not have a DistributedTensor rule registered."
                )
            # default to local tensor ops, this is wrong
            # but we use it now to enable more tensor property access
            else:
                rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
                rs = wrap(rs, args_mesh[0], args[0].placements)
                return rs

    @classmethod
    def from_local(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        run_check: bool = True,
    ) -> "Tensor":
        """
        Create a :class:`spmd.Tensor` from a local torch.Tensor on each rank
        according to the `device_mesh` and `placements` specified.

        Args:
            local_tensor (torch.Tensor): local torch.Tensor on each rank.
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                tensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the placements that
                describes how to place the local torch.Tensor on DeviceMesh, must
                have the same number of elements as `device_mesh.ndim`.
            run_check (bool, optional): indicate whether to run check across ranks
                to check meta information and data. if have :class:`Replicate` in
                `placements`, the data on first rank of the device mesh dimension
                will be broadcasted to other ranks.

        Returns:
            A :class:`spmd.Tensor` object
        """
        # if same shape/dtype, no need to run_check, if not, must allgather
        # the metadatas to check the size/dtype across ranks
        # There should be no data communication unless there's replication
        # strategy, where we broadcast the replication from the first rank
        # in the mesh dimension
        device_mesh = (
            get_global_device_mesh() if device_mesh is None else device_mesh
        )
        # convert the local tensor to desired device base on device mesh's device_type
        local_tensor = local_tensor.to(device_mesh.device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]

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
                if run_check:
                    # broadcast rank 0 tensor to all ranks
                    # only broadcast if run_check is True
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
        placements: Optional[Sequence[Placement]] = None,
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

        # pyre-fixme[16]: `Redistribute` has no attribute `apply`.
        return Redistribute.apply(self, device_mesh, placements)

    def local_tensor(self) -> torch.Tensor:
        return self._local_tensor  # type: ignore

    @property
    def placements(self) -> Sequence[Placement]:
        # placement should be a read only propety
        # to disallow caller modification on it
        # caller who want a different PlacementSpec
        # should call redistribute instead.
        return self._placements

    @property
    def device_mesh(self) -> DeviceMesh:
        # device_mesh should be a read only propety
        # to disallow caller modification on it
        # caller who want a different device_mesh
        # should call redistribute instead.
        return self._device_mesh
