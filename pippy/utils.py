# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import socket
import logging

# Pinning process to a separate GPU if not yet done by launch script
# Notes:
# 1. Needed to work around the issue of RPC not automatically pinning spawned worker threads to CUDA device of the main
# thread
# 2. Must be done before `import torch` at which point CUDA context may be created
# 3. Currently this is enabled by default (as long as #1 is not implemented in RPC). Users may set `PIPPY_PIN_DEVICE` to
# 0 to disable the pinning
if os.getenv("PIPPY_PIN_DEVICE", "1") == "1":
    cuda_devices_str = os.getenv("CUDA_VISIBLE_DEVICES")
    if (
        cuda_devices_str is None  # not set
        or len(cuda_devices_str.split(",")) > 1
    ):  # or set to all devices
        # If launchers like Torchrun sets `LOCAL_RANK`, we would use this information
        local_rank_str = os.getenv("LOCAL_RANK")
        if local_rank_str is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = local_rank_str
            print(
                f"Pinning local process {local_rank_str} to gpu {os.getenv('CUDA_VISIBLE_DEVICES')}"
            )

import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc


VERBOSE = bool(int(os.environ.get("VERBOSE", False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)


def has_efa() -> bool:
    try:
        import subprocess

        return (
            subprocess.run(
                ["fi_info", "-p", "efa", "-t", "FI_EP_RDM"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except FileNotFoundError:
        return False
    except PermissionError:
        return False


def tp_transports():
    return ["shm", "uv"] if has_efa() else None


def run_pippy(run_master, args, *extra_args):
    if not hasattr(args, "world_size"):
        assert hasattr(args, "pp_group_size")
        args.dp_group_size = (
            args.dp_group_size if hasattr(args, "dp_group_size") else 1
        )
    else:
        if not hasattr(args, "dp_group_size"):
            args.pp_group_size = (
                args.pp_group_size
                if hasattr(args, "pp_group_size")
                else args.world_size
            )
            assert args.world_size % args.pp_group_size == 0
            args.dp_group_size = args.world_size // args.pp_group_size
        elif not hasattr(args, "pp_group_size"):
            args.dp_group_size = (
                args.dp_group_size if hasattr(args, "dp_group_size") else 1
            )
            assert args.world_size % args.dp_group_size == 0
            args.pp_group_size = args.world_size // args.dp_group_size
        else:
            pass
            # TODO: doesn't work for PiPPyTrainingArguments
            # assert args.world_size == args.dp_group_size * args.pp_group_size

    actual_world_size = args.dp_group_size * args.pp_group_size
    print(
        f"[PiPPy] World size: {actual_world_size}, "
        f"DP group size: {args.dp_group_size}, "
        f"PP group size: {args.pp_group_size}"
    )

    if args.rank == -1:
        mp.spawn(
            run_worker,
            args=(run_master, args, *extra_args),
            nprocs=actual_world_size,
            join=True,
        )
    elif args.rank < actual_world_size:
        run_worker(args.rank, run_master, args, *extra_args)
    else:
        print("I'm unused, exiting")


def run_worker(rank, run_master, args, *extra_args):
    args.rank = rank

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    actual_world_size = args.dp_group_size * args.pp_group_size

    # TODO: Move to training args, blocked by: cannot pickle 'TensorPipeRpcBackendOptions' object
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=512, rpc_timeout=1800, _transports=tp_transports()
    )
    if args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
            for i in range(actual_world_size):
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
            # Does not seem effective for RPC device pinning. TODO
            # options.set_devices([f'cuda:{dev_id}'])
        else:
            args.cuda = 0
            print("Warning: no CUDA device found. Running on CPU instead.")

    args.device = f"cuda:{dev_id}" if args.cuda else "cpu"
    print(
        f"rank = {rank} host/pid/device = "
        f"{socket.gethostname()}/{os.getpid()}/{args.device}"
    )

    # Init DDP process group
    backend = "nccl" if args.cuda else "gloo"
    torch.distributed.init_process_group(
        backend=backend, rank=rank, world_size=actual_world_size
    )

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=actual_world_size,
        rpc_backend_options=options,
    )

    global dp_pg_per_pp_rank
    dp_ranks_per_pp_rank = (
        torch.arange(actual_world_size)
        .reshape(args.pp_group_size, args.dp_group_size)
        .tolist()
    )
    dp_pg_per_pp_rank = [
        torch.distributed.new_group(ranks) for ranks in dp_ranks_per_pp_rank
    ]

    pp_ranks_per_dp_group = [
        [i * args.dp_group_size + rank for i in range(args.pp_group_size)]
        for rank in range(args.dp_group_size)
    ]

    args.driver_group = torch.distributed.new_group(
        list(range(args.dp_group_size))
    )

    global exclude_master
    exclude_master = (
        args.exclude_master if hasattr(args, "exclude_master") else 0
    )

    if rank >= 0 and rank // args.dp_group_size == 0:
        args.driver_index = rank
        args.local_driver_index = os.getenv("LOCAL_RANK", rank)
        run_master(pp_ranks_per_dp_group[rank], args, *extra_args)
    rpc.shutdown()
