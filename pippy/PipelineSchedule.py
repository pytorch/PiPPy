# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
SINGLE HOST:

python PipelineSchedule.py

or

with torchrun (1x2, 1 host with 2 processes):
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 --nnodes=1 --nproc-per-node=2 PipelineSchedule.py

MULTIPLE HOSTS:

torchrun --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR --nnodes=$NUM_NODES --nproc-per-node=$NUM_TRAINERS PipelineSchedule.py

e.g. (2x2, 2 hosts with 2 processes)
torchrun --rdzv-backend=c10d --rdzv-endpoint=node1.example.com:29400 --nnodes=2 --nproc-per-node=2 PipelineSchedule.py
"""
import argparse
import logging
import os

from collections import deque
from datetime import timedelta
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

logger = logging.getLogger(__name__)


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

    def forward(self, x):
        a = self.wi(x)
        a = self.wh1(a)
        a = self.wh2(a)
        a = self.wh3(a)
        b = self.gelu_act(a)
        c = self.wo(b)
        return c


class PipelineStage(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        rank: int,
        world_size: int,
        meta_input: torch.Tensor,
    ):
        super().__init__()
        self.rank = rank
        self.stage_id = stage_id
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1
        self.num_stages = num_stages
        # When we materialize the model partition on cuda, we call reset_parameters() if it is available
        self.module = module.to(device=torch.cuda.current_device())
        if hasattr(self.module, "reset_parameters"):
            with torch.no_grad():
                self.module.reset_parameters()

        meta_output = self.module(meta_input)
        self.fwd_input = torch.empty_like(meta_input, device="cuda")
        self.fwd_output = None
        self.fwd_output_grads = torch.empty_like(meta_output, device="cuda")
        self.fwd_outputs_for_backward = deque()

        self.prev_stage = (rank - 1) % world_size
        self.next_stage = (rank + 1) % world_size

        self.fwd_recv_queue = None
        self.bwd_recv_queue = None

        self.requests: List[dist.P2POp] = []
        logger.info(
            f"finished pipeline stage init, {self.stage_id=}, {self.is_first_stage=}, {self.is_last_stage=}, {self.num_stages=}, {self.fwd_input.shape=}, {self.fwd_output_grads.shape=}"
        )

    def init_p2p_neighbors(self):
        """
        Set up p2p communitors between previous and next stages
        by sending a dummy tensor.

        If this is used, must be called for all pipeline stages.
        """
        ops = []
        recv_tensor = torch.zeros(1, device="cuda")
        send_tensor = torch.ones(1, device="cuda")
        # forward
        if not self.is_first_stage:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.prev_stage))
        if not self.is_last_stage:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.next_stage))

        # backward
        if not self.is_first_stage:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.prev_stage))
        if not self.is_last_stage:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.next_stage))

        return True

    def get_fwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        return [dist.P2POp(dist.irecv, self.fwd_input, self.prev_stage)]

    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        if self.is_last_stage:
            return []
        return [dist.P2POp(dist.isend, self.fwd_output, self.next_stage)]

    def forward(self, input_data, is_first_mb, is_last_mb):
        logger.info(
            f"[{self.rank} FORWARD {self.stage_id}] is_first_mb {is_first_mb} is_last_mb {is_last_mb}"
        )
        if self.is_first_stage:
            self.fwd_input = input_data

        # this is needed when we access the gradients for this in backward()
        self.fwd_input.requires_grad = True
        self.fwd_input.retain_grad()

        # perform forward pass on module
        self.fwd_output = self.module(self.fwd_input)

        output_for_backward = (
            self.compute_loss() if self.is_last_stage else self.fwd_output
        )

        # we store a ref to the input/output pair for this forward to be later used by the corresponding backward
        self.fwd_outputs_for_backward.append(
            (self.fwd_input, output_for_backward)
        )

        return self.fwd_output

    def get_bwd_send_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        assert self.fwd_input.grad is not None, "grad must be valid"
        return [dist.P2POp(dist.isend, self.fwd_input.grad, self.prev_stage)]

    def get_bwd_recv_ops(self) -> Optional[dist.P2POp]:
        if self.is_last_stage:
            return []
        return [dist.P2POp(dist.irecv, self.fwd_output_grads, self.next_stage)]

    def sync_recv_backward_inputs(self) -> None:
        ops = self.get_bwd_recv_ops()
        if ops:
            dist.batch_isend_irecv(ops).pop().wait()

    def _wait_backward_inputs(self):
        assert (
            self.bwd_recv_queue is not None
        ), "Waiting for backward input without enqueueing one"
        self.bwd_recv_queue.wait()
        self.bwd_recv_queue = None
        return self.fwd_output_grads

    def backward(self, is_first_mb, is_last_mb):
        logger.info(
            f"[{self.rank} BACKWARD {self.stage_id}] is_first_mb {is_first_mb} is_last_mb {is_last_mb}"
        )

        if self.is_last_stage:
            fwd_inputs, loss = self.fwd_outputs_for_backward.popleft()
        else:
            fwd_inputs, fwd_outputs = self.fwd_outputs_for_backward.popleft()

        # Compute gradients
        if self.is_last_stage:
            torch.autograd.backward(loss, retain_graph=True)
        else:
            torch.autograd.backward(
                fwd_outputs, self.fwd_output_grads, retain_graph=True
            )

        return fwd_inputs

    def compute_loss(self):
        if self.fwd_output is None:
            raise RuntimeError("forward() must be called before compute_loss()")
        # TODO: use a real loss function passed in
        return self.fwd_output.mean()


def setup(local_rank, world_size):
    # If this is a child process (i.e., its PID is not the same as the PID of the process that started this script)
    if os.getppid() != os.getpid():
        set_up_logging(local_rank)

    # initialize the process group
    logger.info(f"init for rank {local_rank}")
    dist.init_process_group("nccl")
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)

    logger.info(f"finish init for rank {local_rank}")


class PipelineScheduleGPipe:
    def __init__(self, stage):
        self._stage = stage

    def step(self, microbatches):
        for i, mb in enumerate(microbatches):
            with record_function(f"Forward {i}"):
                is_last_mb = i == len(microbatches) - 1

                ops = self._stage.get_fwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                self._stage.forward(
                    mb, is_first_mb=i == 0, is_last_mb=is_last_mb
                )

                ops = self._stage.get_fwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

                logger.info(
                    f"{self._stage.stage_id} forward {i} finished, microbatch: {mb.shape}"
                )

        for i, _ in enumerate(microbatches):
            with record_function(f"Backward {i}"):

                ops = self._stage.get_bwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                self._stage.backward(
                    is_first_mb=i == 0,
                    is_last_mb=i == len(microbatches) - 1,
                )

                ops = self._stage.get_bwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

            logger.info(f"{self._stage.stage_id} backward {i} finished")


class PipelineScheduleLoopedBFS:
    def __init__(self, stages):
        self._stages = stages

    def step(self, microbatches):
        for s, stage in enumerate(self._stages):
            for i, mb in enumerate(microbatches):
                with record_function(f"Stage {s} Forward"):
                    is_last_mb = i == len(microbatches) - 1

                    ops = stage.get_fwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    stage.forward(mb, is_first_mb=i == 0, is_last_mb=is_last_mb)

                    ops = stage.get_fwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)

        for stage in reversed(self._stages):
            for i in range(len(microbatches)):
                with record_function(f"Stage {stage.stage_id} Backward"):
                    ops = stage.get_bwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    stage.backward(
                        is_first_mb=i == 0,
                        is_last_mb=i == len(microbatches) - 1,
                    )

                    ops = stage.get_bwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)


class PipelineScheduleLoopedDFS:
    def __init__(self, stages, n_microbatch, pp_id, n_pp):
        assert (
            n_microbatch % n_pp == 0
        ), f"Looped DFS schedule requires microbatch_size ({n_microbatch}) to be a multiple of n_pp ({n_pp})"

        self.stages = stages
        self.n_microbatch = n_microbatch

        self.n_stages = len(stages)
        self.total_stages = self.n_stages * n_pp
        # world_size
        self.n_pp = n_pp

        self.stage_id_to_global_stage_id = [
            (i * n_pp) + pp_id for i in range(self.n_stages)
        ]

        # pp_id is the same as local rank within the PP dimension
        self.pp_id = pp_id

        # number of sequences (chunks) to divide microbatches into == microbatch_size / (microbatch_size / n_pp)
        self.seq_size = n_pp

        # warmup steps for latest pp stage is trivial to compute
        # increment warmup_steps by 2 for each hop away
        self.warmup_steps = (len(stages) - 1) * self.seq_size
        self.warmup_steps += 2 * ((n_pp - 1) - pp_id)
        self.forward_steps = len(stages) * n_microbatch
        self.total_steps = self.warmup_steps + (len(stages) * n_microbatch)
        logger.info(
            f"pp_id {pp_id} warmup_steps {self.warmup_steps} forward_steps {self.forward_steps} total_steps {self.total_steps}"
        )

    def step(self, microbatches):
        """
        # n_loop = n_stage / n_pp
        # run microbatches in sequences of NPp

        schedule operates at the rank level

        highest rank has a warmup (F only) count of [len(stages) - 1] * seq_size
        each hop away from highest rank adds 2 warmup stages
            - one happened before highest rank's warmup started,
            - one waiting for backward result to trickle down from highest rank
        dist_from_highest = (worldsize - 1) - rank

        total_steps = warmup_steps + (num_stages * num_microbatch)


        Rank 0: 0F 0F 0F 0F 2F 2F 2F 2F
        Rank 1:    1F 1F 1F 1F 3F3B 3F 3F 3F
        """

        def minibatch_index(step):
            # Given the step index, find the corresponding minibatch index.

            # equivalent to a triple nested loop like this
            # for sequence_id in range(self.seq_size):
            #     for stage in self.stages:
            #         for microbatch_within_sequence:
            #             ...
            # step: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            # index:0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4,  5,  6,  7,  6,  7
            return (step % self.seq_size) + self.seq_size * int(
                step / (self.seq_size * self.n_stages)
            )

        def stage_index(step):
            # step: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            # index:0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,  1,  0,  0,  1,  1
            return int((step / self.seq_size) % self.n_stages)

        """

        my theory was that the hang could be fixed if I orchestrate the recvs after the sends from the schedule side, but i should probably
        see if i can prove what caused the hang before i work on it further
        """
        logger.info(
            f"rank {self.pp_id} - minibatch_index {[minibatch_index(step) for step in range(self.total_steps)]}"
        )
        logger.info(
            f"rank {self.pp_id} - stage_index {[stage_index(step) for step in range(self.total_steps)]}"
        )

        forward_batched_op_handles = []
        backward_batched_op_handles = []

        # edge case for first stage on each rank we need to call receive, recv for future microbatches will be fetched after fwd
        forward_first_recv = self.stages[0].get_fwd_recv_ops()

        # edge case for the last stage on each rank we need to call receive, recv for future microbatches will be fetched after bwd
        backward_first_recv = self.stages[-1].get_bwd_recv_ops()

        backward_stages = list(reversed(self.stages))
        for step in range(self.total_steps):
            mb_id_fwd = minibatch_index(step)
            fwd_stage_id = stage_index(step)
            forward_stage = self.stages[fwd_stage_id]
            fwd_stage_id_next = None
            forward_stage_next = None

            backward_step = step - self.warmup_steps
            mb_id_bwd = minibatch_index(backward_step)
            bwd_stage_id = stage_index(backward_step)
            bwd_stage_id_next = None
            backward_stage_next = None
            backward_stage = backward_stages[bwd_stage_id]

            # info for next stages
            if step < self.total_steps:
                fwd_stage_id_next = stage_index(step + 1)
                forward_stage_next = self.stages[fwd_stage_id_next]
                bwd_stage_id_next = stage_index(backward_step + 1)
                backward_stage_next = backward_stages[bwd_stage_id_next]

            if step < self.forward_steps:

                if forward_first_recv:
                    logger.info(
                        f"rank {self.pp_id} - forward edge case for first stage"
                    )
                    dist.batch_isend_irecv(forward_first_recv).pop().wait()
                    forward_first_recv = None

                if forward_batched_op_handles:
                    assert (
                        len(forward_batched_op_handles) == 1
                    ), "only support one batched op at a time"
                    logger.info(
                        f"rank: {self.pp_id} - waiting on batched_op_handles before fwd"
                    )
                    forward_batched_op_handles.pop().wait()

                with record_function(f"Stage {forward_stage.stage_id} Forward"):
                    logger.info(
                        f"pp_id {self.pp_id} step {step} forward_stage {forward_stage.stage_id} mb_id {mb_id_fwd}"
                    )
                    forward_stage.forward(
                        microbatches[mb_id_fwd],
                        is_first_mb=mb_id_fwd == 0,
                        is_last_mb=mb_id_fwd == len(microbatches) - 1,
                    )

                requests: List[dist.P2POp] = []

                # send output activations if this is not the last stage
                ops = forward_stage.get_fwd_send_ops()
                requests.extend(ops)

                # add recv for the NEXT stage, do not do this for last stage
                if fwd_stage_id_next is not None:
                    ops = forward_stage_next.get_fwd_recv_ops()
                    if mb_id_fwd != len(microbatches) - 1:
                        requests.extend(ops)

                if requests:
                    logger.info(
                        f"rank: {self.pp_id}, current stage_id {self.stage_id_to_global_stage_id[fwd_stage_id]}, next stage_id {self.stage_id_to_global_stage_id[fwd_stage_id]} requests - {[(req.op, req.peer) for req in requests]}"
                    )
                    forward_batched_op_handles.append(
                        dist.batch_isend_irecv(requests).pop()
                    )

            if step >= self.warmup_steps:
                if backward_first_recv:
                    logger.info(
                        f"rank {self.pp_id} - backward edge case for last stage"
                    )
                    dist.batch_isend_irecv(backward_first_recv).pop().wait()
                    backward_first_recv = None

                if backward_batched_op_handles:
                    assert (
                        len(backward_batched_op_handles) == 1
                    ), "only support one batched op at a time"
                    logger.info(
                        f"rank: {self.pp_id} - waiting on batched_op_handles before bwd"
                    )
                    backward_batched_op_handles.pop().wait()

                with record_function(
                    f"Stage {backward_stage.stage_id} Backward"
                ):
                    logger.info(
                        f"pp_id {self.pp_id} step {step}/{self.total_steps} backward_step {backward_step} backward_stage_id {backward_stage.stage_id} mb_id {mb_id_bwd}"
                    )
                    backward_stage.backward(
                        is_first_mb=mb_id_bwd == 0,
                        is_last_mb=mb_id_bwd == len(microbatches) - 1,
                    )

                requests: List[dist.P2POp] = []

                # send bwd grad if this is not the first stage
                ops = backward_stage.get_bwd_send_ops()
                requests.extend(ops)

                # add recv for the NEXT stage, do not do this for first stage
                if bwd_stage_id_next is not None:
                    ops = backward_stage_next.get_bwd_recv_ops()
                    if mb_id_bwd != len(microbatches) - 1:
                        requests.extend(ops)

                if requests:
                    logger.info(
                        f"rank: {self.pp_id}, current stage_id {self.stage_id_to_global_stage_id[bwd_stage_id]}, next stage_id {self.stage_id_to_global_stage_id[bwd_stage_id_next]} requests - {[(req.op, req.peer) for req in requests]}"
                    )
                    backward_batched_op_handles.append(
                        dist.batch_isend_irecv(requests).pop()
                    )

        logger.info("Step exiting")


def main(**kwargs):
    torch.manual_seed(42)
    print(f"MY KWARGS ARE {kwargs}")
    rank = kwargs["rank"]
    local_rank = kwargs["local_rank"]
    world_size = kwargs["world_size"]

    setup(local_rank, world_size)
    logger.info(
        f"====== World Rank {rank}, Local Rank {local_rank}, World Size {world_size} main ======"
    )

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

    stage_model = PipelineStage(
        module_list[rank], rank, world_size, rank, world_size, x
    )
    stage_model.init_p2p_neighbors()

    stage_model_looped = [
        PipelineStage(
            module_list[rank],
            stage_id=(world_size * i) + rank,
            num_stages=world_size * world_size,
            rank=rank,
            world_size=world_size,
            meta_input=x,
        )
        for i in range(world_size)
    ]
    x_cuda_empty = torch.empty_like(x, device="cuda")
    microbatches = [
        torch.randn_like(x_cuda_empty) for _ in range(n_microbatches)
    ]

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

        logger.info(f"====== Rank {rank} profile ======")

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        #    with record_function(schedule):
        pipeline.step(microbatches)

        # TODO - default should be no profiling.
        """if not kwargs["no_trace"]:
            trace_dir = kwargs["trace_dir"]
            if not os.path.exists(trace_dir):
                os.mkdir(trace_dir)
            prof.export_chrome_trace(f"{trace_dir}/{schedule}_rank{rank}_trace.json")
        """
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

    class FstringFormatter(logging.Formatter):
        def format(self, record):
            return f"[{rank}][{record.levelname}][{self.formatTime(record)}][{os.path.basename(__file__)}:{record.lineno}]: {record.getMessage()}"

    formatter = FstringFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if __name__ == "__main__":
    rank = os.environ.get("RANK", None)
    local_rank = os.environ.get("LOCAL_RANK", None)
    world_size = os.environ.get("WORLD_SIZE", None)
    master_addr = os.environ.get("MASTER_ADDR", None)
    master_port = os.environ.get("MASTER_PORT", None)

    parser = argparse.ArgumentParser(description="Pipeline Stages Runner")
    parser.add_argument("--no_trace", action="store_true")
    parser.add_argument("--trace_dir", type=str, default="./traces")
    parser.add_argument(
        "--schedules",
        type=str,
        nargs="+",
        choices=["gpipe", "looped_bfs", "looped_dfs"],
        default=["gpipe", "looped_bfs"],
    )
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

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
        n_gpus = 4
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
