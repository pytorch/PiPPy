# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import math
from typing import List, cast

import torch
from torch.utils._pytree import tree_map, tree_flatten
from typing import Dict, Callable, Optional, Sequence
from spmd.tensor.device_mesh import get_global_device_mesh, DeviceMesh
from spmd.tensor.placement_types import (
    Placement,
    Shard,
    Replicate,
    _Partial,
    PlacementSpec,
)
from spmd.tensor.redistribute import Redistribute

from spmd.tensor.utils import (
    unwrap_local_tensor,
    unwrap_mesh,
    unwrap_spec,
    wrap,
)
from spmd.tensor.dispatch import OpInfo, dispatch_operator


class DTensor(torch.Tensor):  # pyre-ignore[13]: pyre is bad at __new__
    _local_tensor: torch.Tensor
    _placement_spec: PlacementSpec
    __slots__ = ["_local_tensor", "_placement_spec"]

    # class attribute that handles operator placements propagation
    # rules, keyed by aten op name, value is propagation func
    _op_to_rules: Dict[str, Callable] = {}

    # class attribute that handles custom registered ops, all handled
    # custom ops should appear in this table, and overriding the default
    # operators
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    _custom_dispatch_ops: Dict[str, Callable] = {}

    @staticmethod
    def __new__(
        cls,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        size: torch.Size,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> "DTensor":
        # new method instruct wrapper tensor and add placement spec
        # it does not do actual distribution, __init__ should do it instead.
        # TODO: implement __init__ for tensor constructors
        assert isinstance(placements, list)
        # sizes = _flatten_tensor_size(size)
        dtype = kwargs["dtype"]
        layout = kwargs["layout"]
        requires_grad = kwargs["requires_grad"]
        strides = kwargs["strides"]
        device = kwargs.get("device", "cpu")

        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            size,
            strides=strides,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=requires_grad,
        )
        # deepcopy and set spec, data should be handled
        # by __init__ or from_local instead.
        r._placement_spec = PlacementSpec(
            r.ndim, device_mesh, copy.deepcopy(placements)
        )
        return r

    # pyre-fixme[14]: `__repr__` overrides method defined in `Tensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor({self._local_tensor}, placements={self._placements})"

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        # if we find nn.functional name in dispatch op, dispatch to it instead,
        # this allow us to override some python level behaviors that wouldn't be
        # possible in __torch_dispatch__ level.
        if func.__name__ in DTensor._custom_dispatch_ops:
            # dispatch to the same table as the name should be different between
            # torch_function and torch_dispatch
            return DTensor._dist_tensor_dispatch_ops[func.__name__](
                *args, **kwargs
            )
        else:
            # if not, just do nothing here
            return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(f"?? func: {func}")
        # check that we are not getting mixed vanilla and Distributed tensors
        arg_list, arg_spec = tree_flatten(args)
        for arg in arg_list:
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                raise RuntimeError(
                    f"{func}: got mixed distributed and non-distributed tensors."
                )

        # User defined custom aten operator implementation
        if str(func) in DTensor._custom_dispatch_ops:
            # dispatch to user defined custom distributed tensor ops
            return DTensor._custom_dispatch_ops[str(func)](*args, **kwargs)

        # unwrap local tensors, and placement specs, then
        # call into dispatch logic and get back local tensor
        # results and output placement specs
        arg_spec = tree_map(unwrap_spec, args)
        args_with_local_tensors = tree_map(unwrap_local_tensor, args)
        kwargs_spec = tree_map(unwrap_spec, kwargs)
        kwarg_with_local_tensors = tree_map(unwrap_local_tensor, kwargs)

        op_info = OpInfo(
            func,
            args_with_local_tensors,
            kwarg_with_local_tensors,
            arg_spec,
            kwargs_spec,
        )

        return dispatch_operator(op_info, DTensor._op_to_rules)

    @classmethod
    def from_local(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        run_check: bool = True,
    ) -> "DTensor":
        """
        Create a :class:`DTensor` from a local torch.Tensor on each rank
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
            A :class:`DTensor` object
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
            strides=local_tensor.stride(),
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=local_tensor.requires_grad,
        )
        dist_tensor._local_tensor = local_tensor
        return dist_tensor

    def to_local(self) -> torch.Tensor:
        return self._local_tensor  # type: ignore

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
    ) -> "DTensor":
        # This API perform necessary transformations and get
        # a new DTensor with the new spec. i.e. for
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

    @property
    def placements(self) -> Sequence[Placement]:
        # placement should be a read only propety
        # to disallow caller modification on it
        # caller who want a different PlacementSpec
        # should call redistribute instead.
        return self._placement_spec.placements

    @property
    def device_mesh(self) -> DeviceMesh:
        # device_mesh should be a read only propety
        # to disallow caller modification on it
        # caller who want a different device_mesh
        # should call redistribute instead.
        return self._placement_spec._device_mesh

    # TODO: This is a temporary hack to unblock TP efforts. We need to
    # come up with a more principle design for customized ops like this.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _view_with_sharding_dim_change(self, sharding_dim, shape):
        if (
            self.placements[0].is_shard(dim=sharding_dim)
            or self.placements[0].is_replicate()
            or self.placements[0].is_partial()
        ):
            return self.view(shape)
        else:
            if sharding_dim < 0:
                sharding_dim += self.dim()

            device_mesh = self.device_mesh
            world_size = device_mesh.size(dim=0)
            new_sharding_placement = [Shard(sharding_dim)]

            # Fix shape
            try:
                infer_idx = shape.index(-1)
            except ValueError:
                infer_idx = None

            # Infer the dim which is specified with -1.
            if infer_idx is not None:
                st_size = math.prod(self.size())  # type: ignore[attr-defined]
                shape_size = -1 * math.prod(shape)  # type: ignore[attr-defined]
                # pyre-fixme[60]: Concatenation not yet support for multiple variadic
                shape = (
                    *shape[:infer_idx],
                    st_size // shape_size,
                    *shape[infer_idx + 1 :],
                )

            # pyre-fixme[60]: Concatenation not yet support for multiple variadic
            new_local_tensor_size = (
                *shape[:sharding_dim],
                shape[sharding_dim] // world_size,
                *shape[sharding_dim + 1 :],
            )
            new_local_tensor = self.to_local().view(*new_local_tensor_size)

            return DTensor.from_local(
                new_local_tensor, device_mesh, new_sharding_placement
            )
