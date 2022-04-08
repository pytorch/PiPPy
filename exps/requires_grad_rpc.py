#! torchrun --standalone --nnodes=1 --nproc_per_node=2
# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.distributed.rpc as rpc

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

def print_requires_grad(v):
    print(f'Rank 1: requires_grad {v.requires_grad} grad_fn {v.grad_fn} is_leaf {v.is_leaf}')


if local_rank == 0:
    x = torch.relu(torch.randn(5, 3, requires_grad=True))
    print(f'Rank 0: requires_grad {x.requires_grad} grad_fn {x.grad_fn} is_leaf {x.is_leaf}')
    rpc.rpc_sync(to=1, func=print_requires_grad, args=(x,))

rpc.shutdown()

"""
Rank 0: requires_grad True grad_fn <ReluBackward0 object at 0x7fdf7eb79eb0> is_leaf False
Rank 1: requires_grad True grad_fn None is_leaf True
"""
