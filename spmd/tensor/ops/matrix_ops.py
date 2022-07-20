# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import torch.utils._pytree as pytree
from torch.distributed.distributed_c10d import ReduceOp
from spmd.tensor.api import Tensor
from spmd.tensor.placement_types import (
    Shard,
    Replicate,
    _Partial,
)
from spmd.tensor.ops.utils import (
    unwrap_local_tensor,
    unwrap_single_placement,
    register_impl,
)


@register_impl("aten.addmm.default")
def dist_addmm(input: Tensor, mat1: Tensor, mat2: Tensor, beta=1, alpha=1) -> Tensor:
    # dist addmm:
    # input:shard(0)    mat1: shard(0),  mat2: replicate
    # input:shard(1)    mat1: replicate, mat2: shard(1)
    # input:replicate   mat1: shard(0),  mat2: replicate
    # input:replicate   mat1: replicate, mat2: shard(1)
    # input:replicate   mat1: shard(0),  mat2: shard(1)
    local_input, local_mat1, local_mat2 = pytree.tree_map(
        unwrap_local_tensor, (input, mat1, mat2)
    )
    input_placement, mat1_placement, mat2_placement = pytree.tree_map(
        unwrap_single_placement, (input, mat1, mat2)
    )
    device_mesh = mat1.device_mesh

    assert input_placement.is_replicate(), "only support replication now"

    # only implemented combo with no comm for now
    # TODO: implement all combinations
    if mat1_placement.is_shard(dim=0) and mat2_placement.is_replicate():
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, device_mesh, mat1.placements)
    elif mat1_placement.is_replicate() and mat2_placement.is_shard(dim=1):
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, device_mesh, mat2.placements)
    elif mat1_placement.is_replicate() and mat2_placement.is_replicate():
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(
            local_res, device_mesh, mat1.placements, run_check=False
        )
    else:
        raise RuntimeError(f"addmm operator supported for inputs: {mat1}, {mat2}")


@register_impl("aten.mm.default")
def dist_mm(mat1: Tensor, mat2: Tensor) -> Tensor:
    # dist mm:
    # mat1: shard(0),  mat2: replicate
    # mat1: replicate, mat2: shard(1)
    # mat1: shard(1),  mat2: shard(0)
    # mat1: shard(0),  mat2: shard(1)
    local_mat1, local_mat2 = pytree.tree_map(unwrap_local_tensor, (mat1, mat2))
    mat1_placement, mat2_placement = pytree.tree_map(
        unwrap_single_placement, (mat1, mat2)
    )
    device_mesh = mat1.device_mesh

    # only implemented the first 3
    # TODO: implement all combinations
    if mat1_placement.is_shard(dim=0) and mat2_placement.is_replicate():
        local_res = local_mat1.mm(local_mat2)
        return Tensor.from_local(local_res, device_mesh, mat1.placements)
    elif mat1_placement.is_replicate() and mat2_placement.is_shard(dim=1):
        local_res = local_mat1.mm(local_mat2)
        return Tensor.from_local(local_res, device_mesh, mat2.placements)
    elif mat1_placement.is_shard(dim=1) and mat2_placement.is_shard(dim=0):
        local_res = local_mat1.mm(local_mat2)
        placements = [_Partial(ReduceOp.SUM)]
        partial_sum = Tensor.from_local(local_res, device_mesh, placements)
        # all reduce across ranks
        replicate_placements = [Replicate()]
        return partial_sum.redistribute(device_mesh, replicate_placements)
    else:
        raise RuntimeError(f"mm operator supported for inputs: {mat1}, {mat2}")


@register_impl("aten.t.default")
def dist_t(self: Tensor) -> Tensor:
    # transpose with sharding
    local_mat = pytree.tree_map(unwrap_local_tensor, self)
    assert local_mat.ndim == 2
    mat_placement = pytree.tree_map(unwrap_single_placement, self)
    transposed_local_mat = local_mat.t()
    device_mesh = self.device_mesh

    new_shard_dim = 1 if mat_placement.is_shard(dim=0) else 0
    return Tensor.from_local(transposed_local_mat, device_mesh, [Shard(new_shard_dim)])
