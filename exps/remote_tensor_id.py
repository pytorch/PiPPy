#! torchrun --standalone --nnodes=1 --nproc_per_node=2
import torch
import torch.distributed.rpc as rpc

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)


def print_tensor_id(t):
    print(f'Rank {local_rank} tensor id {id(t)}')

if local_rank == 0:
    x = torch.randn(5, 3, requires_grad=True)

    rpc.remote(1, print_tensor_id, (x,))
    rpc.remote(1, print_tensor_id, (x,))
    print_tensor_id(x)
    rpc.remote(0, print_tensor_id, (x,))

rpc.shutdown()
