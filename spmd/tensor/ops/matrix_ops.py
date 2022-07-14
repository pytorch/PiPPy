# implement matrix related ops for distributed tensor
import torch
import torch.utils._pytree as pytree
from torch.distributed.distributed_c10d import (
    ReduceOp
)
from spmd.tensor.api import Tensor
from spmd.tensor.placement_types import (
    Shard,
    Replicate,
    _Partial
)
from spmd.tensor.ops.utils import (
    unwrap_local_tensor,
    unwrap_single_placement,
    is_shard_on_dim,
    register_impl
)


@register_impl("aten.addmm.default")
def dist_addmm(input: Tensor, mat1: Tensor, mat2: Tensor, beta=1, alpha=1) -> Tensor:
    # dist addmm:
    # input:shard(0)    mat1: shard(0),  mat2: replicate
    # input:shard(1)    mat1: replicate, mat2: shard(1)
    # input:replicate   mat1: shard(0),  mat2: replicate
    # input:replicate   mat1: replicate, mat2: shard(1)
    # input:replicate   mat1: shard(0),  mat2: shard(1)
    local_input, local_mat1, local_mat2 = pytree.tree_map(unwrap_local_tensor, (input, mat1, mat2))
    input_placement, mat1_placement, mat2_placement = pytree.tree_map(unwrap_single_placement, (input, mat1, mat2))
    device_mesh = mat1.device_mesh
    world_size = device_mesh.size()
    current_rank = device_mesh.get_rank()

    assert isinstance(input_placement, Replicate), "only support replication now"
    
    # only implemented combo with no comm for now
    # TODO: implement all combinations
    if isinstance(mat1_placement, Shard) and isinstance(mat2_placement, Replicate):
        mat1_shard_dim = mat1_placement.dim
        chunk_size = mat1.size(0) // world_size
        assert mat1_shard_dim == 0, "shard dim should be 0!"
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, device_mesh, mat1.placements)
    elif isinstance(mat1_placement, Replicate) and isinstance(mat2_placement, Shard):
        mat2_shard_dim = mat2_placement.dim
        assert mat2_shard_dim == 1, "shard dim should be 1!"
        chunk_size = mat1.size(1) // world_size
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, device_mesh, mat2.placements)
    elif isinstance(mat1_placement, Replicate) and isinstance(mat2_placement, Replicate):
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, device_mesh, mat1.placement, run_check=False)
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
    mat1_placement, mat2_placement = pytree.tree_map(unwrap_single_placement, (mat1, mat2))
    device_mesh = mat1.device_mesh

    # print(f"?????!!! sharded mm mat1 size: {mat1.size()}, mat2 size: {mat2.size()}")
    # print(f"?????!!! sharded mm mat1 placement {mat1_placement}, mat2 placement: {mat2_placement}")

    # only implemented the first 3
    # TODO: implement all combinations
    if is_shard_on_dim(mat1_placement, 0) and isinstance(mat2_placement, Replicate):
        local_res = local_mat1.mm(local_mat2)
        return Tensor.from_local(local_res, device_mesh, mat1.placements)
    elif isinstance(mat1_placement, Replicate) and is_shard_on_dim(mat2_placement, 1):
        local_res = local_mat1.mm(local_mat2)
        return Tensor.from_local(local_res, device_mesh, mat2.placements)
    elif is_shard_on_dim(mat1_placement, 1) and is_shard_on_dim(mat2_placement, 0):
        local_res = local_mat1.mm(local_mat2)
        placements = [_Partial(ReduceOp.SUM)]
        partial_sum = Tensor.from_local(local_res, device_mesh, placements)
        # all reduce across ranks
        placements[0] = Replicate()
        return partial_sum.redistribute(device_mesh, placements)
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

    new_shard_dim = 1 if is_shard_on_dim(mat_placement, 0) else 0
    return Tensor.from_local(transposed_local_mat, device_mesh, [Shard(new_shard_dim)])
