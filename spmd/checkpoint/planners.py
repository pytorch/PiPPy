import copy
import dataclasses
from functools import reduce, partial
import io
import math
from pickle import FALSE
from typing import Dict,Any, List, Sequence, Tuple
from torch.distributed._shard.checkpoint.planner import LoadPlan, ReadItem, SavePlan
from torch.distributed._shard.sharded_tensor import shard

from torch.distributed._shard.sharded_tensor.api import ShardedTensor
import torch.distributed as dist
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint.metadata import BytesStorageMetadata, ChunkStorageMetadata, Metadata, MetadataIndex, STATE_DICT_TYPE, TensorStorageMetadata
from torch.distributed._shard.checkpoint.planner_helpers import _create_sharded_read_items, _create_read_items
from torch.distributed._shard.checkpoint.utils import find_state_dict_object, find_tensor_shard
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata, TensorProperties
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.remote_device import _remote_device

from spmd import DTensor as DT
from torch.distributed._shard.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner
)
from torch.distributed._shard.api import _shard_tensor
import logging

from .nested_dict import (
    flatten_state_dict,
    set_element,
    get_element,
)
logger = logging.getLogger(__file__)

def check_box_overlap(box0, box1):
    """
    Checks if two boxes overlap. Tuples are (offset, lengths)
    """

    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(box0.offsets)
    for i in range(ndims):
        if box0.offsets[i] >= box1.offsets[i] + box1.sizes[i]:
            return False
        if box1.offsets[i] >= box0.offsets[i] + box0.sizes[i]:
            return False

    return True

def check_box_bounds(outer_box_size, inner_box):
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False

    return True

def validate_global_plan(global_plan: List[SavePlan], metadata: Metadata) -> bool:
    all_good = True
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        #check for overlap
        chunks_volume = 0
        for chunk_idx, chunk0 in enumerate(value.chunks):
            if not check_box_bounds(value.size, chunk0):
                logger.warning(f"key:{key} has out of bounds chunk: tensor-size:{value.size} chunk: {chunk0}")
                all_good = False
            chunks_volume += math.prod(chunk0.sizes)

            for chunk1 in value.chunks[chunk_idx + 1:]:
                if check_box_overlap(chunk0, chunk1):
                    logger.warning(f"key:{key} has overlapping chunks: {chunk0} {chunk1}")
                    all_good = False

        tensor_volume = math.prod(value.size)
        if chunks_volume != tensor_volume:
            logger.warning(f"key:{key} invalid fill tensor-volume: {tensor_volume} chunks-volume: {chunks_volume}")
            all_good = False

    return all_good

# TODO find a better name for this class
class UberSavePlanner(DefaultSavePlanner):
    def __init__(
        self,
        flatten_state_dict=True,
        flatten_sharded_tensors=True,
        dedup_replicated_tensors=True
    ):
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.dedup_replicated_tensors = dedup_replicated_tensors

    def init(self, state_dict: Dict[str, Any], is_coordinator: bool) -> None:
        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)
        if self.flatten_sharded_tensors:
            state_dict = flatten_sharded_tensors(state_dict)
        return super().init(state_dict, is_coordinator)

    def create_local_plan(self) -> SavePlan:
        plan = super().create_local_plan()
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        return plan

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        if self.dedup_replicated_tensors:
            all_plans = dedup_tensors(all_plans)

        global_plan, metadata = super().create_global_plan(all_plans)

        if self.flatten_state_dict:
            merged_mappings = reduce(lambda x, y: x | y, (p.planner_data for p in global_plan))
            metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

        if not validate_global_plan(global_plan, metadata):
            raise ValueError("Failed to validate global plan")

        return global_plan, metadata

# TODO find a better name for this class
class UberLoadPlanner(DefaultLoadPlanner):
    def __init__(
        self,
        flatten_state_dict=True,
        flatten_sharded_tensors=True,
    ):
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors

    def init(self, state_dict: STATE_DICT_TYPE, metadata: Metadata, is_coordinator: bool) -> None:
        if self.flatten_sharded_tensors:
            state_dict = flatten_sharded_tensors(state_dict)

        self.original_state_dict = state_dict        
        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)
            # Note that we don't need the checkpoint mappings as we assume
            # that they are compatible with the one we just computed

        return super().init(state_dict, metadata, is_coordinator)

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        set_element(
            self.original_state_dict,
            self.mappings[read_item.dest_index.fqn],
            torch.load(value)
        )

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        obj = get_element(self.original_state_dict, self.mappings[index.fqn])
        return find_tensor_shard(obj, index)
