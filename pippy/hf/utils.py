# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from transformers import (
    TrainingArguments
)
from transformers.utils import (
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_tpu_available,
)
from transformers.utils import torch_required, cached_property

logger = logging.getLogger(__name__)


@dataclass
class PiPPyTrainingArguments(TrainingArguments):
    dp_group_size: int = field(
        default=1, metadata={"help": "DDP group size."}
    )

    pp_group_size: int = field(
        default=8, metadata={"help": "Pipeline group size."}
    )

    rank: int = field(
        default=int(os.getenv("RANK", -1)), metadata={"help": "Rank."}
    )

    local_rank: int = field(
        default=-1, metadata={"help": "Local Rank."}
    )

    local_process_index: int = field(
        default=-1, metadata={"help": "Local process index."}
    )

    process_index: int = field(
        default=-1, metadata={"help": "Process index."}
    )

    master_addr: str = field(
        default=os.getenv('MASTER_ADDR', 'localhost'), metadata={"help": "Master address."},
    )

    master_port: str = field(
        default=os.getenv('MASTER_PORT', '29500'), metadata={"help": "Master port."},
    )

    exclude_master: int = field(
        default=0, metadata={"help": "Exclude master.", "choices": [0, 1]},
    )

    # TODO: use `no_cuda` instead?
    cuda: int = field(
        default=int(torch.cuda.is_available()), metadata={"help": "Exclude master.", "choices": [0, 1]},
    )

    chunks: Optional[int] = field(
        default=None, metadata={"help": "Number of Chunks."}
    )

    record_mem_dumps: int = field(
        default=0, metadata={"help": "Record memory dumps flag."}
    )

    checkpoint: int = field(
        default=1, metadata={"help": "Checkpoint flag."}
    )

    # @staticmethod
    # def _has_efa() -> bool:
    #     try:
    #         import subprocess
    #         return subprocess.run(["fi_info", "-p", "efa", "-t", "FI_EP_RDM"],
    #                               stdout=subprocess.DEVNULL,
    #                               stderr=subprocess.DEVNULL).returncode == 0
    #     except FileNotFoundError:
    #         return False
    #
    # @staticmethod
    # def _tp_transports():
    #     return ["shm", "uv"] if PiPPyTrainingArguments._has_efa() else None

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.cuda:
            n_devs = torch.cuda.device_count()
            if n_devs > 0:
                dev_id = self.rank % n_devs
                return torch.device(f'cuda:{dev_id}')
            else:
                self.cuda = 0
                return torch.device('cpu')
        else:
            return torch.device('cpu')

    @property
    def world_size(self):
        return self.dp_group_size

    def __post_init__(self):
        super().__post_init__()
        # TODO: checks here

        # TODO: cannot pickle 'TensorPipeRpcBackendOptions' object
        # self.options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256,
        #                                                rpc_timeout=1800,
        #                                                _transports=PiPPyTrainingArguments._tp_transports())

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        if is_torch_available() and self.world_size > 1:
            main_process_desc = "main process"
            if local:
                is_main_process = self.local_process_index == 0
                main_process_desc = "main local process"
            elif is_sagemaker_mp_enabled():
                is_main_process = False  # TODO is_main_process = smp.rank() == 0
            else:
                is_main_process = self.process_index == 0

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")
                    if is_torch_tpu_available():
                        pass  # TODO xm.rendezvous(desc)
                    elif is_sagemaker_dp_enabled():
                        pass  # TODO dist.barrier()
                    else:
                        torch.distributed.barrier(group=dp_pg_for_reference)
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    if is_torch_tpu_available():
                        pass  # TODO xm.rendezvous(desc)
                    elif is_sagemaker_dp_enabled():
                        pass  # TODO dist.barrier()
                    else:
                        torch.distributed.barrier(group=dp_pg_for_reference)
        else:
            yield


def has_efa() -> bool:
    try:
        import subprocess
        return subprocess.run(["fi_info", "-p", "efa", "-t", "FI_EP_RDM"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        return False
    except PermissionError:
        return False


def tp_transports():
    return ["shm", "uv"] if has_efa() else None


def run_pippy(model_args, data_args, training_args, run_master):
    actual_world_size = training_args.dp_group_size * training_args.pp_group_size
    if training_args.rank == -1:
        mp.spawn(run_worker, args=(model_args, data_args, training_args, run_master), nprocs=actual_world_size,
                 join=True)
    elif training_args.rank < actual_world_size:
        run_worker(training_args.rank, model_args, data_args, training_args, run_master)
    else:
        print("I'm unused, exiting")


def run_worker(rank, model_args, data_args, training_args, run_master):
    os.environ['MASTER_ADDR'] = training_args.master_addr
    os.environ['MASTER_PORT'] = training_args.master_port

    actual_world_size = training_args.dp_group_size * training_args.pp_group_size

    # TODO: Move to training args, blocked by: cannot pickle 'TensorPipeRpcBackendOptions' object
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256,
                                              rpc_timeout=1800,
                                              _transports=tp_transports())
    if training_args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
            for i in range(actual_world_size):
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})

    print(f"rank = {rank} host/pid/device = "
          f"{socket.gethostname()}/{os.getpid()}/{training_args.device}")

    # Init DDP process group
    backend = "nccl" if training_args.cuda else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=actual_world_size)

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=actual_world_size,
        rpc_backend_options=options
    )

    global dp_pg_per_pp_rank
    dp_ranks_per_pp_rank = torch.arange(actual_world_size).reshape(training_args.pp_group_size,
                                                                   training_args.dp_group_size).tolist()
    dp_pg_per_pp_rank = [torch.distributed.new_group(ranks) for ranks in dp_ranks_per_pp_rank]

    pp_ranks_per_dp_group = [[i * training_args.dp_group_size + rank for i in range(training_args.pp_group_size)]
                             for rank in range(training_args.dp_group_size)]

    global dp_pg_for_reference
    dp_pg_for_reference = torch.distributed.new_group(list(range(training_args.dp_group_size)))

    global exclude_master
    exclude_master = training_args.exclude_master

    if rank >= 0 and rank // training_args.dp_group_size == 0:
        training_args.rank = rank
        training_args.local_rank = -1  # TODO: must be -1 to disable automatic DDP in the HF trainer
        training_args.process_index = rank  # TODO: Is it correct?
        training_args.local_process_index = rank  # TODO: Is it correct?
        run_master(model_args, data_args, training_args, pp_ranks_per_dp_group[rank])
    rpc.shutdown()
