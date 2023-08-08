import argparse
import os
import time

import torch
import torch.distributed as dist

import torch.nn as nn

from pippy.compile import compile_stage

from pippy.IR import annotate_split_points, PipeSplitWrapper
from pippy.microbatch import sum_reducer, TensorChunkSpec

from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)

from torch.profiler import profile, ProfilerActivity


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=1024)

    # distributed
    parser.add_argument(
        "--backend", type=str, default="nccl"
    )  # 'nccl', 'gloo', etc.
    parser.add_argument("--rank", type=int, default=int(os.environ["RANK"]))
    parser.add_argument(
        "--world_size", type=int, default=int(os.environ["WORLD_SIZE"])
    )
    parser.add_argument(
        "--device", type=str, default=f"cuda:{os.environ['LOCAL_RANK']}"
    )

    parser.add_argument("--train_iters", type=int, default=10)
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--i_stage", type=int, default=1)
    parser.add_argument("--n_chunks", type=int, default=2)

    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--inner_cut", dest="inner_cut", action="store_true")
    parser.add_argument("--inference", dest="inference", action="store_true")

    args = parser.parse_args()

    return args


def get_rand(args):
    x = torch.rand(
        (args.batch_size, args.hidden_size),
        device=args.device,
    )
    y = torch.rand(
        (args.batch_size, args.hidden_size),
        device=args.device,
    )
    return x, y


class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super(MLPModule, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid, bias=False)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid, bias=False)

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


class ExampleCode(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlps = nn.ModuleDict(
            dict(
                m=nn.ModuleList(
                    [MLPModule(args.hidden_size) for _ in range(args.n_layer)]
                ),
            )
        )
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, y=None):
        for block in self.mlps.m:
            x = block(x)

        if y is not None:
            loss = self.mse_loss(x, y)
        else:
            loss = None

        return x, loss


def tp_mlp(model, mesh, tp_dim=0, mlp="mlp"):
    parallelize_module(
        model, mesh, {mlp: PairwiseParallel()}, tp_mesh_dim=tp_dim
    )

    return model


def even_cut(model, args, pp_size):
    """
    Evenly cut a model into pp_size stages
    """
    cut = {}
    cutpoint = args.n_layer // pp_size
    for i in range(args.n_layer):
        name = f"mlps.m.{i}"
        if i > 0 and i % cutpoint == 0:
            cut[name] = PipeSplitWrapper.SplitPoint.BEGINNING  # or END

    if args.rank == 0:
        print(cut)

    annotate_split_points(model, cut)


def pp_and_tp_selective(
    model, mesh, args, tp_attn_layers=None, tp_mlp_layers=None, cut_fn=even_cut
):
    """
    Apply pipeline parallelism and tensor parallelism to a model.
    """

    pp_dim, tp_dim = 0, 1
    pp_rank, tp_rank = args.rank // args.tp_size, args.rank % args.tp_size
    pp_groups = mesh.get_dim_groups()[pp_dim]

    # TP
    for i in range(args.n_layer):
        mlp = tp_mlp(model, mesh, tp_dim, f"mlps.m.{i}")

    X, Y = get_rand(args)

    # PP
    cut_fn(model, args, args.pp_size * args.i_stage)
    num_stages = args.pp_size * args.i_stage

    if args.inference:
        stage = compile_stage(
            model,
            pp_rank,
            args.pp_size,
            args.n_chunks,
            args.device,
            pp_groups,
            example_inputs=[X, Y],
            num_stages=num_stages,
            schedule="TwoLevel",
        )
    else:
        output_chunk_spec = (TensorChunkSpec(0), sum_reducer)
        stage = compile_stage(
            model,
            pp_rank,
            args.pp_size,
            args.n_chunks,
            args.device,
            pp_groups,
            example_inputs=[X, Y],
            output_chunk_spec=output_chunk_spec,
            num_stages=num_stages,
            schedule="TwoLevel",
        )

    return model, stage


def pp_tp_inference(stage, mesh, args):
    pp_dim, tp_dim = 0, 1
    pp_rank, tp_rank = args.rank // args.tp_size, args.rank % args.tp_size
    pp_groups = mesh.get_dim_groups()[pp_dim]

    train_iters = 10 if args.debug else args.train_iters
    local_iter_num = 0
    iter_time = 0.0
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=5, wait=0, warmup=4, active=1, repeat=1
        ),
    ) as prof:
        while local_iter_num < train_iters:
            t0 = time.perf_counter()
            X, Y = get_rand(args)
            if pp_rank == 0:
                out = stage(X)
            elif pp_rank == args.pp_size - 1:
                out = stage(Y)
            else:
                out = stage()
            t1 = time.perf_counter()
            dt = t1 - t0
            local_iter_num += 1
            iter_time += dt
            prof.step()
            dist.barrier()

    prof.export_chrome_trace(f"trace_rank{args.rank}.json")

    return local_iter_num, iter_time


def pp_tp_train(stage, mesh, args):
    pp_dim, tp_dim = 0, 1
    pp_rank, tp_rank = args.rank // args.tp_size, args.rank % args.tp_size
    pp_groups = mesh.get_dim_groups()[pp_dim]

    train_iters = 10 if args.debug else args.train_iters
    optimizer = torch.optim.AdamW(
        stage.submod.parameters(), lr=args.learning_rate
    )
    local_iter_num = 0
    iter_time = 0.0
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=5, wait=0, warmup=4, active=1, repeat=1
        ),
    ) as prof:
        while local_iter_num < train_iters:
            optimizer.zero_grad()
            t0 = time.perf_counter()
            X, Y = get_rand(args)
            if pp_rank == 0:
                out = stage(X)
            elif pp_rank == args.pp_size - 1:
                out = stage(Y)
            else:
                out = stage()
            optimizer.step()
            t1 = time.perf_counter()
            dt = t1 - t0
            local_iter_num += 1
            iter_time += dt
            prof.step()

    prof.export_chrome_trace(f"trace_rank{args.rank}.json")

    return local_iter_num, iter_time


if __name__ == "__main__":
    _multi_gpu = int(os.environ.get("RANK", -1)) != -1  # verify distributed run
    assert (
        _multi_gpu
    ), "this config assumes distributed setup - multi-gpu not ready here."

    args = get_args()

    device_type = (
        "cuda" if "cuda" in args.device else "cpu"
    )  # for later use in torch.autocast
    torch.cuda.set_device(args.device)

    dist.init_process_group(
        backend=args.backend, rank=args.rank, world_size=args.world_size
    )

    torch.manual_seed(args.seed)

    twod_mesh = DeviceMesh(
        device_type=device_type,
        mesh=torch.arange(0, args.world_size).view(-1, args.tp_size),
    )

    model = ExampleCode(args)
    model.to(args.device)

    ref_x, ref_y = get_rand(args)

    model, stage = pp_and_tp_selective(model, twod_mesh, args)

    if args.inference:
        iter_count, iter_time = pp_tp_inference(stage, twod_mesh, args)
    else:
        iter_count, iter_time = pp_tp_train(stage, twod_mesh, args)

    if args.rank == 0:
        print(f"\nInference demo completed. Check your trace!\n")

    dist.destroy_process_group()
