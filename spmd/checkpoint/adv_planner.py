import dataclasses
from functools import reduce
import io
import math
import logging
from typing import List, Tuple

import torch
from torch.distributed.checkpoint.planner import ReadItem, SavePlan

from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)

from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
)

from torch.distributed.checkpoint.utils import find_tensor_shard

from .traverse import set_element, get_element
from .nested_dict import FLATTEN_MAPPING, flatten_state_dict
from .nested_tensor import flatten_sharded_tensors
from .dedup_tensors import dedup_tensors

logger: logging.Logger = logging.getLogger(__file__)


def _check_box_overlap(
    box0: ChunkStorageMetadata, box1: ChunkStorageMetadata
) -> bool:
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


def _check_box_bounds(
    outer_box_size: torch.Size, inner_box: ChunkStorageMetadata
) -> bool:
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False

    return True


def _validate_global_plan(
    global_plan: List[SavePlan], metadata: Metadata
) -> bool:
    all_good = True
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks_volume = 0
        for chunk_idx, chunk0 in enumerate(value.chunks):
            if not _check_box_bounds(value.size, chunk0):
                logger.warning(
                    f"key:{key} has out of bounds chunk: tensor-size:{value.size} chunk: {chunk0}"
                )
                all_good = False
            chunks_volume += math.prod(chunk0.sizes)

            for chunk1 in value.chunks[chunk_idx + 1 :]:
                if _check_box_overlap(chunk0, chunk1):
                    logger.warning(
                        f"key:{key} has overlapping chunks: {chunk0} {chunk1}"
                    )
                    all_good = False

        tensor_volume = math.prod(value.size)
        if chunks_volume != tensor_volume:
            logger.warning(
                f"key:{key} invalid fill tensor-volume: {tensor_volume} chunks-volume: {chunks_volume}"
            )
            all_good = False

    return all_good


class AdvSavePlanner(DefaultSavePlanner):
    """
    SavePlanner that adds multiple features on top of DefaultSavePlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    dedup_replicated_tensors: Remove duplicated shards, for when using DT with Replicated placement.
    """

    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_replicated_tensors: bool = True,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.dedup_replicated_tensors = dedup_replicated_tensors
        self.mappings = {}

    def init(self, state_dict: STATE_DICT_TYPE, is_coordinator: bool) -> None:
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

    def create_global_plan(
        self, all_plans: List[SavePlan]
    ) -> Tuple[List[SavePlan], Metadata]:
        if self.dedup_replicated_tensors:
            all_plans = dedup_tensors(all_plans)

        global_plan, metadata = super().create_global_plan(all_plans)

        if self.flatten_state_dict:
            merged_mappings = reduce(
                lambda x, y: x | y, (p.planner_data for p in global_plan)
            )
            metadata = dataclasses.replace(
                metadata, planner_data=merged_mappings
            )

        if not _validate_global_plan(global_plan, metadata):
            raise ValueError("Failed to validate global plan")

        return global_plan, metadata


class AdvLoadPlanner(DefaultLoadPlanner):
    """
    LoadPlanner that adds multiple features on top of DefaultLoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    """

    original_state_dict: STATE_DICT_TYPE
    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.original_state_dict = {}
        self.mappings = {}

    def init(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
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
            torch.load(value),
        )

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        obj = get_element(self.original_state_dict, self.mappings[index.fqn])
        assert isinstance(obj, torch.Tensor)
        return find_tensor_shard(obj, index)
