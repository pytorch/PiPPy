import torch

import torch.distributed.rpc as rpc

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

class TupleFactory:
    def give_tuple(self, x):
        return (x, 1337)

if local_rank == 0:
    tup_factory_rrefs = []

    for rank in range(1, world_size):
        tup_factory_rrefs.append(rpc.remote(rank, TupleFactory))

    tuple_rrefs = []
    for rref in tup_factory_rrefs:
        tuple_rrefs.append(rref.remote().give_tuple(3))

    print('Tuple RRefs')
    print([rref.owner() for rref in tuple_rrefs])

    selected_item_rrefs = []

    for tuple_rref in tuple_rrefs:
        selected_item_rrefs.append(tuple_rref.remote().__getitem__(1))

    print('Selected item RRef owners')
    print([rref.owner() for rref in selected_item_rrefs])

    values = [rref.to_here() for rref in selected_item_rrefs]
    assert all(value == 1337 for value in values)


rpc.shutdown()
