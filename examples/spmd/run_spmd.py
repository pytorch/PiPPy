import logging
import os
import random
from copy import deepcopy
from typing import Literal, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from spmd import Schema, SPMD
from spmd.compiler.graph_optimization import GraphOptimization
from spmd.tensor import DeviceMesh, Replicate
from torch.nn.parallel import DistributedDataParallel as DDP


logger: logging.Logger = logging.getLogger(__name__)

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def setup(rank: int, world_size: int) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def teardown(rank: int) -> None:

    # Wait for all ranks to reach here before starting shutdown.
    dist.barrier()
    dist.destroy_process_group()


class ReplicaModel(nn.Module):
    def __init__(self, layer_count: int = 2, _with_bias: bool = False) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(10, 10, bias=_with_bias) for _ in range(layer_count)]
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Literal[0]]:
        return sum([self.seq(x)])


def work_main(rank: int, world_size: int) -> None:
    # must randomize the seed per GPU to ensure all_reduce
    # is meaningful.
    torch.manual_seed(rank)

    _device_type = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_placement = torch.arange(
        world_size
    )
    mesh = DeviceMesh(device_type=_device_type, mesh=gpu_placement)
    
    # control depth of ReplicaModel
    layers = 4

    model = ReplicaModel(layer_count=layers).to(_device_type)

    ddp = DDP(deepcopy(model))
    ddp.to(rank)

    run_backward = True
    all_spmd = []
    optimizations = [
        [GraphOptimization("fuse_communication_ring")],
        [GraphOptimization("fuse_communication_cat")],
    ]
    for optim in optimizations:
        spmd = SPMD(
            deepcopy(model),
            schema=Schema(
                mesh=DeviceMesh(
                    _device_type, gpu_placement
                ),
                placements=[Replicate()],
            ),
            optimize_first_iter=True,
            optimizations=optim,
        )
        all_spmd.append(spmd)

    x = torch.randn(2, 10).to(_device_type)
    
    # fire off comms
    for spmd in all_spmd:
        spmd(x).sum().backward()
    ddp(x).sum().backward()

    if rank == 0:
        for spmd in all_spmd:
            for i, (p1, p2) in enumerate(
                zip(ddp.parameters(), spmd.parameters())
            ):
                # just show first row of first 2 grad tensors for quick visual
                if i < 2:
                    # visual display of initial grads
                    div_grad = p2.grad[0] / world_size

                    print(f"DDP:\n {p1.grad[0]}\nSPMD:\n {div_grad}\n")  # type: ignore

                assert p1.grad.allclose(  # type: ignore
                    p2.grad / world_size
                ), "Mismatch in resulting grads between DDP and SPMD."

def main(rank: int, world_size: int) -> None:

    setup(rank, world_size)
    _world_size = dist.get_world_size()
    
    # main work
    work_main(rank, world_size)

    # teardown
    teardown(rank)


if __name__ == "__main__":
    # Note: this only works on a single node.
    os.environ["MASTER_ADDR"] = "localhost"
    
    port = random.randint(49152, 65535)
    os.environ["MASTER_PORT"] = str(port)

    world_size = 2
    assert torch.cuda.is_available(), "GPUs are needed to run this example!"
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
