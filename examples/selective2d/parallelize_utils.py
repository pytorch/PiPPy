# Copyright (c) Meta Platforms, Inc. and affiliates
import time

import torch
import torch.distributed as dist

from torch.profiler import profile, ProfilerActivity


def pp_tp_inference(stage, mesh, args, data_fn):
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
            X, Y = data_fn(args)
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


def pp_tp_train(stage, mesh, args, data_fn):
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
            X, Y = data_fn(args)
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
