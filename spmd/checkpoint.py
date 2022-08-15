import io
from dataclasses import replace
from typing import Any, Dict

import torch
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed._shard.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed._shard.checkpoint.metadata import (
    STATE_DICT_TYPE,
    Metadata,
    MetadataIndex,
)
from torch.distributed._shard.checkpoint.planner import ReadItem, SavePlan
from torch.distributed._shard.sharded_tensor.api import ShardedTensor


def get_ranks(pg: ProcessGroup):
    """
    Return an array of global ranks for a given process group.
    """
    return [c10d._get_global_rank(pg, i) for i in range(pg.size())]


class ProcessGroupAwareSavePlanner(DefaultSavePlanner):
    """
    ProcessGroupAwareSavePlanner extends DefaultSavePlanner and re-write the state_dict.
    """

    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        """
        Rename all keys of sharded tensors from sub-process groups by prefixing it 
        with a PG specific string.
        """
        state_dict_copy = {}
        for k, v in state_dict.items():
            # Only rename the fqn if the current process group is a sub-process group.
            if (
                isinstance(v, ShardedTensor)
                and state_dict[k]._process_group
                != dist.distributed_c10d._get_default_group()
            ):
                pg_global_ranks = get_ranks(state_dict[k]._process_group)
                fqn = "_".join([str(rank) for rank in pg_global_ranks]) + "_" + k
                state_dict_copy[fqn] = state_dict[k]
            else:
                state_dict_copy[k] = v

        super().init(state_dict_copy, is_coordinator)


class ProcessGroupAwareLoadPlanner(DefaultLoadPlanner):
    """
    ProcessGroupAwareSaveLoader extends DefaultLoadPlanner and re-write the state_dict.
    """

    def init(
        self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool
    ) -> None:
        """
        Rename all keys of sharded tensors from sub-process groups by prefixing it 
        with a PG specific string.
        """
        self.original_state_dict = state_dict

        state_dict_copy = {}
        for k, v in state_dict.items():
            # Only rename the fqn if the current process group is a sub-process group.
            if (
                isinstance(v, ShardedTensor)
                and state_dict[k]._process_group
                != dist.distributed_c10d._get_default_group()
            ):
                pg_global_ranks = get_ranks(state_dict[k]._process_group)
                fqn = "_".join([str(rank) for rank in pg_global_ranks]) + "_" + k
                state_dict_copy[fqn] = state_dict[k]
            else:
                state_dict_copy[k] = ""

        super().init(state_dict_copy, metadata, is_coordinator)

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        """
        This method makes sure that the non sharded_tensor value of the original_state_dict
        also gets loaded properly.
        """
        self.original_state_dict[read_item.dest_index.fqn] = torch.load(value)

