import torch
import spmd.tensor.api as spmd_tensor
from spmd.tensor.placement_types import (
    Shard,
    Replicate,
    _Partial
)


def redistribute_spmd_tensor(input, device_mesh, placements):
    current_placements = input.placements
    if isinstance(current_placements[0], Shard) and isinstance(placements[0], Replicate):
        assert len(placements) == 1, "Only support 1-d placement for now"
        assert input.device_mesh.mesh.equal(device_mesh.mesh), "cross mesh comm not support yet"
        # record in ctx
        # for shard, all_gather all shards and return the global tensor
        global_tensor = torch.empty(
            input.size(),
            device=input._local_tensor.device,
            dtype=input.dtype
        )
        # NOTE: all_gather_base only works well when tensor
        # sharded on a sequential list of devices
        device_mesh.all_gather_base(global_tensor, input._local_tensor)
        replica_tensor = spmd_tensor.Tensor.from_local(global_tensor, device_mesh, placements)
        replica_tensor._placements[0] = Replicate()
        return replica_tensor
    elif isinstance(current_placements[0], _Partial) and isinstance(placements[0], Replicate):
        reduced_tensor = device_mesh.all_reduce(input._local_tensor, current_placements[0].reduce_op)
        replica_tensor = spmd_tensor.Tensor.from_local(reduced_tensor, device_mesh, current_placements)
        # change placement to replicate
        replica_tensor._placements[0] = Replicate()
        return replica_tensor
    elif isinstance(current_placements[0], Replicate) and isinstance(placements[0], Shard):
        # slice the tensor to local shard then return shard tensor
        local_tensor = input.local_tensor()
        shard_dim = placements[0].dim
        num_chunks = device_mesh.size()
        assert input.size(shard_dim) % num_chunks == 0, "Only support chunk sharding evenly now"
        chunk_size = input.size(shard_dim) // num_chunks
        my_rank = device_mesh.get_rank()
        local_shard = local_tensor[my_rank * chunk_size, (my_rank + 1) * chunk_size]
        return spmd_tensor.Tensor.from_local(local_shard, device_mesh, placements)
    elif current_placements == placements:
        return input
    else:
        raise RuntimeError(f"Converting from {current_placements} to {placements} not supported!")


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, device_mesh, placements):
        ctx.previous_placement = input.placements
        return redistribute_spmd_tensor(
            input,
            device_mesh,
            placements
        )

    @staticmethod
    def backward(ctx, grad_output):
        previous_placement = ctx.previous_placement
        return redistribute_spmd_tensor(
            input,
            device_mesh,
            previous_placement
        )
