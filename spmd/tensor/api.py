# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import math
import warnings
import torch
from torch.utils._pytree import tree_flatten
from typing import Dict, Callable, Optional, Sequence, Tuple
from spmd.tensor.device_mesh import get_global_device_mesh, DeviceMesh
from spmd.tensor.placement_types import (
    Placement,
    Shard,
    Replicate,
    _Partial,
    PlacementSpec,
)
from spmd.tensor.redistribute import Redistribute

from spmd.tensor.dispatch import operator_dispatch, OpSchema, OutputSharding

# NOTE [Autograd interaction between torch.Tensor]
#
# The autograd functions defined below are being used by the public
# facing APIs (i.e. from_local, to_local) to ensure our DTensor
# works together with torch.Tensor within autograd engine. This
# allows DistributedTensor to exist on part of the module hierarchy
# and still able to calculate gradients across the torch.Tensor and
# DistributedTensor boundary.
# As an example, we have the a module that consists of submodules
# A, B, and C, the execution flow would be like:
#  input(torch.Tensor) -> Module A -> Module B -> Module C -> output (torch.Tensor)
#
# Suppose I only want to make Module B be a sharded module with
# DistributedTensor params, we would need to make the folloing
# flow to work:
#
#  input(torch.Tensor) -> Module A
#       -> DTensor input -> Sharded Module B -> DTensor output
#           -> output (torch.Tensor) -> Module C -> output (torch.Tensor)
#
# We need the conversion from Module A to DTensor input, which is
# `from_local`, and conversion from DTensor output to output, which
# is `to_local`, thus these two functions must be Autograd functions.
#
class ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: "DTensor"):  # type: ignore
        ctx.previous_placement = input.placements
        ctx.previous_device_mesh = input.device_mesh
        return input._local_tensor.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh
        dist_grad = DTensor.from_local(
            grad_output, previous_device_mesh, previous_placement
        )
        return dist_grad


class FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,  # type: ignore
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        run_check: bool,
    ):  # type: ignore
        ctx.previous_placement = placements
        ctx.previous_device_mesh = device_mesh

        if run_check:
            # TODO: by default check tensor metas across rank
            # TODO: See if we need to make this run_check logic
            # have a corresponding backward.
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    # broadcast rank 0 tensor to all ranks
                    # only broadcast if run_check is True
                    device_mesh.broadcast(input, mesh_dim=idx)

        dist_tensor = DTensor(
            input,
            device_mesh,
            placements,
            # requires_grad of the dist tensor depends on if input
            # requires_grad or not
            requires_grad=input.requires_grad,
        )
        return dist_tensor

    @staticmethod
    def backward(ctx, grad_output: "DTensor"):  # type: ignore
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh

        # reshard to the placement when creating DistributedTensor
        # so that the gradient layout matches, and we could return
        # local gradients directly
        if grad_output.placements != previous_placement:
            grad_output = grad_output.redistribute(
                previous_device_mesh, previous_placement
            )

        # TODO: backward is also differentiable now, add a test
        # to test higher level gradients.
        return grad_output.to_local(), None, None, None


def _reshape_alias(
    x: torch.Tensor, shape: Tuple[int, ...], strides: Tuple[int, ...]
) -> torch.Tensor:
    return torch.ops.aten.view(x, shape)


from torch._decomp import decomposition_table

_CURRENT_DECOMPOSITION_TABLE: Dict[torch._ops.OpOverload, Callable] = {
    torch.ops.aten._reshape_alias.default: _reshape_alias,
}


class DTensor(torch.Tensor):  # pyre-ignore[13]: pyre is bad at __new__
    _local_tensor: torch.Tensor
    _placement_spec: PlacementSpec
    __slots__ = ["_local_tensor", "_placement_spec"]

    # class attribute that handles operator placements propagation
    # rules, keyed by aten op name, value is propagation func
    _op_to_rules: Dict[str, Callable[[OpSchema], OutputSharding]] = {}

    # class attribute that handles custom registered ops, all handled
    # custom ops should appear in this table, and overriding the default
    # operators that's been covered by _op_to_rules or fallbacks.
    # (custom operator is the highest priority when dispatching).
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    _custom_dispatch_ops: Dict[str, Callable] = {}

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> "DTensor":
        # TODO: add a docstr about tensor constructor
        # from_local, and distribute_tensor difference.

        # recover tensor shape and strides in the case of sharding
        tensor_shape = list(local_tensor.size())
        tensor_stride = list(local_tensor.stride())
        for idx, placement in enumerate(placements):
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_dim_size = tensor_shape[shard_dim]
                # recover tensor shape on the shard dim
                tensor_shape[shard_dim] = local_dim_size * device_mesh.size(idx)

                # recover tensor stride by modifying the stride that larger than
                # the current stride on the shard_dim
                for i in range(len(tensor_stride)):
                    if tensor_stride[i] > tensor_stride[shard_dim]:
                        tensor_stride[i] = tensor_stride[i] * device_mesh.size(
                            idx
                        )
            elif not isinstance(placement, (Replicate, _Partial)):
                raise RuntimeError(
                    f"placement type {type(placement)} not supported!"
                )

        requires_grad = kwargs.get("requires_grad", False)
        if requires_grad != local_tensor.requires_grad:
            warnings.warn(
                "To construct DTensor from torch.Tensor, it's recommended to "
                "use local_tensor.detach() and make requires_grad consistent."
            )

        # new method instruct wrapper tensor from local_tensor and add
        # placement spec, it does not do actual distribution
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            torch.Size(tensor_shape),
            strides=tensor_stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )
        # deepcopy and set spec, data should be handled
        # by __init__ or from_local instead.
        r._placement_spec = PlacementSpec(
            r.ndim, device_mesh, copy.deepcopy(placements)
        )
        # detach local tensor from autograd graph as we initialize the
        # distributed tensor and autograd will be working on top of
        # the wrapper tensor directly instead of local torch.Tensor
        r._local_tensor = local_tensor.detach()
        return r

    # pyre-fixme[14]: `__repr__` overrides method defined in `DTensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor(local_tensor={self._local_tensor}, device_mesh={self._placement_spec.mesh}, placements={self._placement_spec.placements})"

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
            return DTensor._custom_dispatch_ops[func.__name__](*args, **kwargs)
        else:
            # if not, just do nothing here
            return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        # first we need to lift some private aten aliases to public calls
        if kwargs is None:
            kwargs = {}
        if func in _CURRENT_DECOMPOSITION_TABLE:
            with torch.overrides.enable_reentrant_dispatch():
                return _CURRENT_DECOMPOSITION_TABLE[func](*args, **kwargs)

        # check that we are not getting mixed vanilla and Distributed tensors
        arg_list, _ = tree_flatten(args)
        for arg in arg_list:
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                raise RuntimeError(
                    f"{func}: got mixed distributed and non-distributed tensors."
                )

        if kwargs is None:
            kwargs = {}

        return operator_dispatch(
            func,
            args,
            kwargs,
            DTensor._op_to_rules,
            DTensor._custom_dispatch_ops,
        )

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
                have the same number of elements as `device_mesh.ndim`. If not
                specified, we will by default replicate the tensor across the
                `device_mesh` from the first rank of each dimension of the `device_mesh`.
            run_check (bool, optional): indicate whether to run check across ranks
                to check meta information and data. if have :class:`Replicate` in
                `placements`, the data on first rank of the device mesh dimension
                will be broadcasted to other ranks.

        Returns:
            A :class:`DTensor` object

        .. note:: `from_local` is differentiable, the `requires_grad` of the created
            `DTensor` object will depend on if `local_tensor` requires_grad or not.
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

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor, device_mesh, placements, run_check
        )

    def to_local(self) -> torch.Tensor:
        return ToTorchTensor.apply(self)  # pyre-ignore[16]: autograd func

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

        for placement in placements:
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to _Partial, _Partial is for internal use only!"
                )

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
        return self._placement_spec.mesh

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
