# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import torch.utils._pytree as pytree
from typing import List, Optional
from torch.distributed.distributed_c10d import ReduceOp
from spmd.tensor.api import Tensor
from spmd.tensor.dispatch import OpInfo
from spmd.tensor.placement_types import Shard, Replicate, _Partial, PlacementSpec
from spmd.tensor.utils import (
    unwrap_local_tensor,
)
from spmd.tensor.ops.utils import (
    register_prop_rule,
    register_impl,
)


@register_prop_rule("aten.mm.default")
def mm_rules(op_info: OpInfo) -> Optional[PlacementSpec]:
    # mm propagation rule:
    # mat1: shard(0),  mat2: replicate
    # mat1: replicate, mat2: shard(1)
    # mat1: shard(1),  mat2: shard(0)
    # propagation rules only propagates the combs without communication
    mat1_spec, mat2_spec = op_info.args_spec
    print(f"mat1 spec: {mat1_spec.dims_map}, mat2 spec: {mat2_spec.dims_map}")
    # TODO: support multi-dim device mesh op with einop propagation
    if mat1_spec.placements[0].is_shard(dim=0) and mat2_spec.placements[0].is_replicate():
        return mat1_spec
    elif mat1_spec.placements[0].is_replicate() and mat2_spec.placements[0].is_shard(dim=1):
        return mat2_spec
    elif mat1_spec.placements[0].is_shard(dim=1) and mat2_spec.placements[0].is_shard(dim=0):
        placements = [_Partial(ReduceOp.SUM)]
        return PlacementSpec(mat1_spec.ndim, mat1_spec.mesh, placements)
    else:
        # not local compute, need to rely on auto redistribute, return None
        return None

@register_prop_rule("aten.t.default")
def transpose_rule(op_info: OpInfo) -> Optional[PlacementSpec]:
    mat_spec = op_info.args_spec[0]
    mat_placement = mat_spec.placements[0]
    if not mat_placement.is_shard():
        return mat_spec
    else:
        mat_placement.dim = 1 if mat_placement.is_shard(dim=0) else 0
        return mat_spec


@register_impl("aten.addmm.default")
def dist_addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    # pyre-fixme[2]: Parameter must be annotated.
    beta=1,
    # pyre-fixme[2]: Parameter must be annotated.
    alpha=1,
) -> Tensor:
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
        local_res = local_input.addmm(
            local_mat1, local_mat2, beta=beta, alpha=alpha
        )
        return Tensor.from_local(local_res, device_mesh, mat1.placements)
    elif mat1_placement.is_replicate() and mat2_placement.is_shard(dim=1):
        local_res = local_input.addmm(
            local_mat1, local_mat2, beta=beta, alpha=alpha
        )
        return Tensor.from_local(local_res, device_mesh, mat2.placements)
    elif mat1_placement.is_replicate() and mat2_placement.is_replicate():
        local_res = local_input.addmm(
            local_mat1, local_mat2, beta=beta, alpha=alpha
        )
        return Tensor.from_local(
            local_res, device_mesh, mat1.placements, run_check=False
        )
    else:
        raise RuntimeError(
            f"addmm operator supported for inputs: {mat1}, {mat2}"
        )
