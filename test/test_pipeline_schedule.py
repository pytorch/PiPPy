# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
SINGLE HOST:

python test_pipeline_schedule.py

or

with torchrun (1x2, 1 host with 2 processes):
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 --nnodes=1 --nproc-per-node=2 test_pipeline_schedule.py

MULTIPLE HOSTS:

torchrun --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR --nnodes=$NUM_NODES --nproc-per-node=$NUM_TRAINERS test_pipeline_schedule.py

e.g. (2x2, 2 hosts with 2 processes)
torchrun --rdzv-backend=c10d --rdzv-endpoint=node1.example.com:29400 --nnodes=2 --nproc-per-node=2 test_pipeline_schedule.py
"""

import argparse
import logging
import os
from contextlib import contextmanager, nullcontext

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pippy.PipelineSchedule import (
    PipelineScheduleGPipe,
    PipelineScheduleLoopedBFS,
    PipelineScheduleLoopedDFS,
    PipelineStageV2Impl,
)
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.profiler import record_function


logger = logging.getLogger(__name__)

_null_context = nullcontext()


# profiling context manager
@contextmanager
def maybe_run_profiler(
    use_profiler, trace_dir, schedule, rank, *args, **kwargs
):
    def trace_handler(prof):
        if rank == 0:
            (f"about to EXPORT traces for {schedule} to {trace_dir}")
        prof.export_chrome_trace(
            f"{trace_dir}/{schedule}_rank{rank}_trace.json"
        )

    if use_profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=trace_handler,
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = nullcontext()
        yield None


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.wi = nn.Linear(dim, hidden_dim, bias=False)
        self.wh1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wh2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wh3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wo = nn.Linear(hidden_dim, out_dim, bias=False)
        self.gelu_act = nn.GELU(approximate="tanh")

    def forward(self, x, arg1):
        a = self.wi(x)
        a = self.wh1(a)
        a = self.wh2(a)
        a = self.wh3(a)
        b = self.gelu_act(a)
        c = self.wo(b)
        return c, arg1


def setup(local_rank, init_process_group=False):
    # If this is a child process (i.e., its PID is not the same as the PID of the process that started this script)
    if os.getppid() != os.getpid():
        set_up_logging(local_rank)

    # initialize the process group (not needed if using device_mesh)
    if init_process_group:
        logger.info(f"init for rank {local_rank}")
        if torch.distributed.is_initialized():
            torch.cuda.set_device(local_rank)
    logger.info(f"finish init for rank {local_rank}")


def main(**kwargs):
    torch.manual_seed(42)
    device = torch.device(kwargs["device"])

    local_rank = kwargs["local_rank"]
    world_size = kwargs["world_size"]

    def rank_print(msg):
        if rank == 0:
            print(f"{msg}")

    # use device mesh for cuda - create a device mesh based on the given world_size.
    device_mesh = None
    rank = None

    if device.type == "cuda":
        device_mesh = init_device_mesh(
            device_type="cuda", mesh_shape=(int(os.environ["WORLD_SIZE"]),)
        )

        rank = device_mesh.get_rank()
        rank_print(f"using {device_mesh=}")

    if not device_mesh:
        rank = kwargs["rank"]

    init_process_group = bool(device_mesh is not None)

    setup(local_rank, init_process_group=init_process_group)

    logger.info(
        f"====== World Rank {rank}, Local Rank {local_rank}, World Size {world_size}, Device {device} main ======"
    )

    rank_print(f"My KWARGS are {kwargs}")

    input_dim = 4000
    hidden_dim = 8000
    output_dim = 4000

    module_list = torch.nn.ModuleList(
        modules=[
            MLP(input_dim, hidden_dim, output_dim) for i in range(world_size)
        ]
    )
    microbatch_size = 8
    global_batch_size = 64
    assert global_batch_size % microbatch_size == 0
    n_microbatches = int(global_batch_size / microbatch_size)
    n_pp = world_size

    x = torch.randn([microbatch_size, input_dim]).to("meta")
    unused = torch.ones((1, 1), device="meta")
    input_args = (x, unused)

    stage_model = PipelineStageV2Impl(
        module_list[rank],
        rank,
        world_size,
        rank,
        world_size,
        input_args,
        device,
    )
    stage_model.init_p2p_neighbors()

    stage_model_looped = [
        PipelineStageV2Impl(
            module_list[rank],
            stage_id=(world_size * i) + rank,
            num_stages=world_size * world_size,
            rank=rank,
            world_size=world_size,
            input_args=input_args,
            device=device,
        )
        for i in range(world_size)
    ]
    x_cuda_empty = torch.empty_like(x, device="cuda")

    microbatches = []
    for i in range(n_microbatches):
        microbatches.append(
            (torch.randn_like(x_cuda_empty), torch.ones((1, 1), device="cuda"))
        )

    _run_profiler = kwargs["profiler"]
    _trace_dir = kwargs["trace_dir"]
    for schedule in kwargs["schedules"]:
        logger.info(f"====== Rank {rank} running schedule {schedule} ======")
        if schedule == "gpipe":
            pipeline = PipelineScheduleGPipe(stage_model)
        elif schedule == "looped_bfs":
            pipeline = PipelineScheduleLoopedBFS(stage_model_looped)
        elif schedule == "looped_dfs":
            pipeline = PipelineScheduleLoopedDFS(
                stage_model_looped,
                n_microbatch=n_microbatches,
                pp_id=rank,
                n_pp=n_pp,
            )

        if _run_profiler:
            logger.info(f"====== Rank {rank} profile ======")

        with maybe_run_profiler(
            _run_profiler, _trace_dir, schedule, rank
        ) as _torch_profiler:
            with record_function(schedule):
                pipeline.step(microbatches)
        logger.info(f"====== Rank {rank} finished {schedule} ======")


def main_wrapper(rank, local_rank, world_size, kwargs):
    rank = int(rank)
    world_size = int(world_size)
    if local_rank is None:
        local_rank = rank
    local_rank = int(local_rank)

    os.environ["RANK"] = str(rank)
    main(rank=rank, local_rank=local_rank, world_size=world_size, **kwargs)


def set_up_logging(rank, log_level=logging.INFO):
    """Set up logging"""
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)

    # TODO: seeing double logging due to global logging setup in
    # - fx/passes/utils/matcher_utils.py

    # class FstringFormatter(logging.Formatter):
    #     def format(self, record):
    #         return f"[{rank}][{record.levelname}][{self.formatTime(record)}][{os.path.basename(__file__)}:{record.lineno}]:{record.getMessage()}"

    # formatter = FstringFormatter()
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)


if __name__ == "__main__":
    rank = os.environ.get("RANK", None)
    local_rank = os.environ.get("LOCAL_RANK", None)
    world_size = os.environ.get("WORLD_SIZE", None)
    master_addr = os.environ.get("MASTER_ADDR", None)
    master_port = os.environ.get("MASTER_PORT", None)

    parser = argparse.ArgumentParser(description="Pipeline Stages Runner")
    parser.add_argument("--profiler", type=bool, default=False)
    parser.add_argument("--trace_dir", type=str, default="./pipeline_traces")
    parser.add_argument(
        "--schedules",
        type=str,
        nargs="+",
        choices=["gpipe", "looped_bfs", "looped_dfs"],
        default=["gpipe", "looped_bfs", "looped_dfs"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    kwargs = vars(args)

    if (
        rank is None
        or local_rank is None
        or world_size is None
        or master_addr is None
    ):
        # single host code path
        master_port = "23456"
        master_addr = "localhost"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        n_gpus = 2
        world_size = n_gpus
        os.environ["WORLD_SIZE"] = str(world_size)
        print(
            f"Torchrun was not used. Spawning {world_size} processes on {master_addr}:{master_port}"
        )
        mp.spawn(
            main_wrapper,
            args=(
                None,
                world_size,
                kwargs,
            ),
            nprocs=world_size,
        )
    else:
        # multihost code path (ran with torchrun)
        main_wrapper(rank, local_rank, world_size, kwargs)
