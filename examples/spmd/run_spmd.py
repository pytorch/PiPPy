import logging
import os
from copy import deepcopy
from functools import partial
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from spmd.compiler.log_utils import rank0_debug
from torch.nn.parallel import DistributedDataParallel as DDP

from spmd import SPMD, Schema
from spmd.tensor import DeviceMesh, Replicate
from typing import List, Union, Literal

logger: logging.Logger = logging.getLogger(__name__)
_debug = partial(rank0_debug, logger)  # type: ignore

# globals --------------

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def setup(rank: int, world_size: int, use_cuda: bool = True) -> None:
    logging.getLogger().setLevel(
        logging.DEBUG if rank == 0 else logging.CRITICAL
    )

    if use_cuda:
        _debug("--> init process group using nccl")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        print(f"--> device set for rank {rank}")
    else:
        _debug("--> init process group using gloo")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def teardown(rank: int) -> None:

    # Wait for all ranks to reach here before starting shutdown.
    _debug(f"rank {rank} entering teardown")
    dist.barrier()
    dist.destroy_process_group()
    _debug(f"shut down process group on rank {rank}")


def formatted_print(
    rank: int, name: str, val: int, rank_only: bool = False
) -> None:
    if rank_only and not rank == 0:
        return
    print(f"{rank} --> {name} = {val}")


# --- model

mnist_dims: List[int] = [28 * 28, 500, 250, 100, 50, 25, 10]


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = nn.Sequential(
            *[
                layer
                for i in range(2)
                for layer in [
                    nn.Linear(mnist_dims[i], mnist_dims[i + 1], bias=True),
                    nn.ReLU(),
                ]
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mod(x)


class Permute(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand((5, 10)))
        self.b = torch.nn.Parameter(torch.rand((5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.permute(0, 2, 1)
        return torch.nn.functional.linear(x_t, self.w, self.b)


class ReplicaModel(nn.Module):
    def __init__(self, layer_count: int = 2, _with_bias: bool = False) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(10, 10, bias=_with_bias) for _ in range(layer_count)]
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Literal[0]]:
        return sum([self.seq(x)])


# ------------ main loop --------------------


def work_main(rank: int, world_size: int) -> None:
    # must randomize the seed per GPU to ensure all_reduce
    # is meaningful.
    torch.manual_seed(rank)

    _device_type = "cuda" if torch.cuda.is_available() else "cpu"

    gpu_placement = torch.arange(
        world_size
    )  # .reshape(2, 2) for world_size = 4
    _debug(f"updated gpu placement = {gpu_placement}")

    mesh = DeviceMesh(device_type=_device_type, mesh=gpu_placement)
    _debug(f"mesh set to {mesh}\n")

    # control depth of ReplicaModel
    layers = 2

    # model = Permute().to(rank)  #
    model = ReplicaModel(layer_count=layers).to(_device_type)

    # ddp = DDP(deepcopy(model))
    # ddp.to(rank)

    run_backward = True
    spmd = SPMD(
        deepcopy(model),
        schema=Schema(
            mesh=DeviceMesh(
                _device_type, gpu_placement  # torch.arange(world_size)
            ),
            placements=[Replicate()],
        ),
        optimize_first_iter=run_backward,
    )

    # model input - need to adjust to match models
    # permute_input = x = torch.randn(2, 10, 40).to("cuda")
    x = torch.randn(2, 10).to(_device_type)
    _debug(f"\ninput tensor, first item = {x[0][0]:.4f}")

    # fire off comms
    spmd(x).sum().backward()
    # ddp(x).sum().backward()

    # if rank == 0:
    #     _debug(f" --> backwards run complete, rank {rank}")

    #     print("Visual of resulting grads:\n")
    #     for i, (p1, p2) in enumerate(zip(ddp.parameters(), spmd.parameters())):
    #         # just show first row of first 2 grad tensors for quick visual
    #         if i < 2:
    #             # visual display of initial grads
    #             div_grad = p2.grad[0] / world_size

    #             print(f"DDP:\n {p1.grad[0]}\nSPMD:\n {div_grad}\n")  # type: ignore

    #         assert p1.grad.allclose(  # type: ignore
    #             p2.grad / world_size
    #         ), "Mismatch in resulting grads between DDP and SPMD."
    # _debug("--> run completed, all grads matching!")


# --------- main above -------------------------


def main(rank: int, world_size: int, use_cuda: bool = True) -> None:

    # init
    setup(rank, world_size, use_cuda)

    _world_size = dist.get_world_size()
    logging.info(f"--> World size = {_world_size}")

    # main work
    work_main(rank, world_size)

    # teardown
    teardown(rank)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    # obtain random port
    port = random.randint(49152, 65535)
    os.environ["MASTER_PORT"] = str(port)

    world_size = 2
    use_cuda: bool = DEVICE_TYPE == "cuda"

    print(f"use_cuda == {use_cuda}, starting run_SPMD...\n")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
