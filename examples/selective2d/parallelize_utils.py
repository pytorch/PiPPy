# Copyright (c) Meta Platforms, Inc. and affiliates
import time

import torch
import torch.distributed as dist

from torch.profiler import profile, ProfilerActivity


def pp_tp_inference(stage, mesh, args, data_fn, model=None):
    pp_dim, tp_dim = 0, 1
    pp_rank, tp_rank = args.rank // args.tp_size, args.rank % args.tp_size
    pp_groups = mesh.get_dim_groups()[pp_dim]

    train_iters = 10 if args.debug else args.train_iters
    local_iter_num = 0
    iter_time = 0.0
    warmup_iters = 0
    ref_inputs, outputs = [], []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=5, wait=0, warmup=4, active=1, repeat=1
        ),
    ) as prof:
        while local_iter_num < train_iters:
            X, Y = data_fn(args)
            ref_inputs.append((X, Y))
            t0 = time.perf_counter()
            if pp_rank == 0:
                out = stage(X)
            elif pp_rank == args.pp_size - 1:
                out = stage(Y)
                outputs.append(out)
            else:
                out = stage()
            dist.barrier()
            t1 = time.perf_counter()
            dt = t1 - t0
            if warmup_iters < args.warmup_iters:
                warmup_iters += 1
            else:
                local_iter_num += 1
                iter_time += dt
            prof.step()

    prof.export_chrome_trace(f"trace_rank{args.rank}.json")

    # verify the result
    if model is not None and pp_rank == args.pp_size - 1:
        it = 0
        for X, Y in ref_inputs:
            ref_out, ref_loss = model(X)
            torch.testing.assert_close(ref_out, outputs[it])
            print(f"[Rank{pp_rank}][GPU{args.rank}] iteration {it} passed")
            it += 1

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
