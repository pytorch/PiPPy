import torch
import torch.distributed.rpc as rpc

def return_future():
    return torch.futures.Future()


import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

if local_rank == 0:
    future_rref = rpc.remote(1, return_future)
    future = future_rref.to_here()

rpc.shutdown()
