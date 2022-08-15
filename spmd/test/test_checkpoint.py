# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import shutil
import tempfile


import torch
import torch.distributed as dist
import torch.distributed._shard.checkpoint as cp
from demo import ProcessGroupAwareLoadPlanner, ProcessGroupAwareSavePlanner
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_utils import run_tests

from _utils import DistTensorTestBase, with_comms

CHECKPOINT_DIR = ""

class MyModule(torch.nn.Module):
    def __init__(self, rank, extra_state):
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.arange(
                start=rank * 4, end=rank * 4 + 4, step=1, dtype=torch.float32
            )
        )
        self.extra_state = extra_state

    @property 
    def extra_state(self):
        return self._extra_state
    
    @extra_state.setter 
    def extra_state(self, new_extra_state):
        self._extra_state = new_extra_state

    def get_extra_state(self):
        return {"extra_state": self._extra_state}

    def set_extra_state(self, extra_state_dict):
        self._extra_state = extra_state_dict["extra_state"]


class TestProcessGroupAwareSavePlanner(DistTensorTestBase):
    @with_comms
    def test_process_group_aware_planner(self):

        model = MyModule(rank=dist.get_rank(), extra_state=0).cuda(dist.get_rank())

        fsdp_0 = dist.new_group(ranks=[0, 2])
        fsdp_1 = dist.new_group(ranks=[1, 3])
        if dist.get_rank() % 2 == 0:
            my_fsdp = fsdp_0
        else:
            my_fsdp = fsdp_1

        model = FSDP(model, process_group=my_fsdp)

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()

            cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=ProcessGroupAwareSavePlanner(),
            )

        model = MyModule(rank=100, extra_state=100).cuda(dist.get_rank())
        model = FSDP(model, process_group=my_fsdp)

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()

            cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=cp.FileSystemReader(path=CHECKPOINT_DIR),
                planner=ProcessGroupAwareLoadPlanner(),
            )
            model.load_state_dict(state_dict)

        tensor_dict = {
            0: torch.tensor([ 0,  1, 10, 11], dtype=torch.float32),
            1: torch.tensor([ 4,  5, 14, 15], dtype=torch.float32),
            2: torch.tensor([ 0,  1, 10, 11], dtype=torch.float32),
            3: torch.tensor([ 4,  5, 14, 15], dtype=torch.float32),
        }

        with FSDP.summon_full_params(model):
            self.assertEqual(tensor_dict[dist.get_rank()], model.param.detach()) 
            self.assertEqual(0, model.extra_state)


if __name__ == "__main__":

    CHECKPOINT_DIR = tempfile.TemporaryDirectory()
    print(f"USING {CHECKPOINT_DIR.name} for checkpoints")
    run_tests()
    CHECKPOINT_DIR.cleanup()
