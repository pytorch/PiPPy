# rework ddp_master to use spmd
import torch
import torch.nn as nn

import os
import torch.multiprocessing as mp

# sys.path.append(../)
from spmd import SPMD, Schema
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.fx.experimental.proxy_tensor import make_fx

# from spmd.testing.common_utils import (  # type: ignore
#    DistTensorTestBase,
#   with_comms,
# )
from spmd.tensor import (
    DeviceMesh,
    Replicate,
)
from copy import deepcopy


import torch.distributed as dist
import logging

# globals --------------

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
g_device_type = DEVICE_TYPE


def setup(rank, world_size, use_cuda=True):
    logging.getLogger().setLevel(
        logging.DEBUG if rank == 0 else logging.CRITICAL
    )

    if use_cuda:
        print(f"init for rank {rank}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        print(f"--> init device for rank {rank}")
        torch.cuda.set_device(rank)
        print(f"device set for rank {rank}")
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def teardown(rank) -> None:

    # Wait for all ranks to reach here before starting shutdown.
    print(f"rank {rank} entering teardown")
    dist.barrier()
    dist.destroy_process_group()
    logging.info(f"shut down process group on rank {rank}")


def formatted_print(rank, name, val, rank_only=False):
    if rank_only and not rank == 0:
        return
    print(f"{rank} --> {name} = {val}")


# --- model

DIMS = [28 * 28, 500, 250, 100, 50, 25, 10]


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.Sequential(
            *[
                layer
                for i in range(2)
                for layer in [
                    nn.Linear(DIMS[i], DIMS[i + 1], bias=True),
                    nn.ReLU(),
                ]
            ]
        )

    def forward(self, x):
        return self.mod(x)


class Permute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.w = torch.nn.Parameter(torch.rand((5, 10)))
        self.b = torch.nn.Parameter(torch.rand((5)))

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        return torch.nn.functional.linear(x_t, self.w, self.b)


class replicaModel(nn.Module):
    def __init__(self, layer_count=2, _with_bias=False):
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(10, 10, bias=_with_bias) for _ in range(layer_count)]
        )
        # self.module_list = nn.ModuleList(
        #    [nn.Linear(10, 10) for _ in range(layer_count)]
        # )

    def forward(self, x):
        return sum([self.seq(x)])
        # return sum([m(x) for m in self.module_list])


def message(rank, msg, title=""):
    print(f"{rank} --> {title} ... {msg}\n")


# ------------ main loop --------------------


def work_main(rank, world_size):
    torch.manual_seed(10)

    # model = MyModel()
    # model.to(rank)
    # message(rank, "model moved to gpu")

    # test tensor
    # local_tensor = torch.ones(world_size).to(rank) * rank
    # message(rank, local_tensor, "local_tensor")

    _device_type = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_placement = torch.arange(world_size)  # .reshape(2, 2)
    # print(f"gpu_placement start = {gpu_placement}")
    gpu_placement = gpu_placement.tolist()
    print(f"updated gpu place = {gpu_placement}")

    mesh = DeviceMesh(device_type=_device_type, mesh=gpu_placement)
    print(f"mesh set to {mesh}\n")
    layers = 2

    # model = Permute().to(rank)  #
    model = replicaModel(layer_count=layers).to("cuda")

    ddp = DDP(deepcopy(model))
    ddp.to(rank)
    spmd = SPMD(
        deepcopy(model),
        schema=Schema(
            mesh=DeviceMesh(
                _device_type, gpu_placement  # torch.arange(world_size)
            ),
            placements=[Replicate()],
        ),
    )

    # spmd.to(_device_type)

    x = torch.randn(2, 10).to("cuda")
    if rank == 0:
        print(f"\ninput tensor, first item = {x[0][0]:.4f}")
    # print(f"spmd mesh = {spmd._schema}")

    # fire off comms
    spmd(x).sum().backward()
    ddp(x).sum().backward()

    if rank == 0:
        print(f"backwards run complete, rank {rank}")

        for p1, p2 in zip(ddp.parameters(), spmd.parameters()):
            # DDP divides gradients by world size to compute average, but
            # _Partial tensor shouldn't do that automatically. Hence explicitly
            # do division here.
            div_grad = p2.grad[0] / world_size
            print(f"loop {p1.grad[0]=}\n, {div_grad }")

            # assert p1.grad.allclose(p2.grad), "p1 p2 grad mismatch"
            assert p1.grad.allclose(p2.grad / world_size)

    return

    # execute traced DeviceMesh communication
    # reduced_tensor_fx = traced_fn(local_tensor)
    # message(rank, reduced_tensor_fx, "reduced_tensor_fx")

    # ----- all gather
    dim_to_subgroups = mesh.get_dim_groups()
    for dim, dim_group in enumerate(dim_to_subgroups):
        dim_group_size = dist.get_world_size(dim_group)
        global_ranks = [
            dist.get_global_rank(dim_group, i) for i in range(dim_group_size)
        ]
        # print(f"{rank} --> global ranks, {global_ranks}")

        all_gather_base_tensor = torch.ones(world_size).to(rank) * rank

        def fn_ag(tensor: torch.Tensor, output_shape):
            return mesh.all_gather(tensor, output_shape, mesh_dim=dim)

        # use a local_tensor + 1 for tracing to make sure that we are not
        # simply replaying recorded tensor value
        traced_fn = make_fx(fn_ag)(
            all_gather_base_tensor + 1, (dim_group_size * 3, 3)
        )
        gathered_tensor = traced_fn(
            all_gather_base_tensor, (dim_group_size * 3, 3)
        )

        exp_tensor = torch.ones(3 * dim_group_size, 3)
        for i in range(len(global_ranks)):
            exp_tensor[i * 3 : (i + 1) * 3] = torch.ones(3, 3) * global_ranks[i]

        print(
            f"{rank} --> test all gather.  exp_tensor = {exp_tensor}\n gathered tensor = {gathered_tensor}"
        )

    # res_num = sum(global_ranks)


# --------- main above -------------------------


def main(rank, world_size, use_cuda=True):

    # import os

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
    os.environ["MASTER_PORT"] = "29502"
    world_size = 2
    use_cuda = DEVICE_TYPE == "cuda"
    print(f"use_cuda == {use_cuda}, starting run_fusion...\n")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
