# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.metadata import (
    STORAGE_TYPES,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    ReadItem,
    SavePlan,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    _create_sharded_read_items,
    _create_write_items,
)
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard

from spmd.tensor import DTensor


def init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    level = logging.DEBUG
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    logger.propagate = False
    return logger


dLogger: logging.Logger = init_logger()


class DistributedTensorSavePlanner(DefaultSavePlanner):
    def __init__(self, dedup_replicated_tensors: bool = True):  # pyre-ignore[3]
        self.dedup_replicated_tensors: bool = dedup_replicated_tensors
        super().__init__()

    def create_local_plan(self) -> SavePlan:
        requests = []
        write_item: List[WriteItem]
        for fqn, obj in self.state_dict.items():  # pyre-ignore[16]
            if isinstance(obj, DTensor):
                write_item = [_create_write_items_for_dtensor(fqn, obj)]
            elif (
                not isinstance(obj, DTensor)
                or self.is_coordinator  # pyre-ignore[16]
            ):
                write_item = _create_write_items(fqn, obj)
            requests += write_item

        return SavePlan(requests)

    def lookup_object(self, index: MetadataIndex) -> Any:  # pyre-ignore[3]
        obj = self.state_dict[index.fqn]  # pyre-ignore[16]
        if isinstance(obj, DTensor):
            return obj.to_local()
        else:
            return super().lookup_object(index)

    def create_global_plan(
        self, all_plans: List[SavePlan]
    ) -> Tuple[List[SavePlan], Metadata]:
        if self.dedup_replicated_tensors:
            all_plans = dedup_tensors(all_plans)

        global_plan, metadata = super().create_global_plan(all_plans)
        return global_plan, metadata


class DistributedTensorLoadPlanner(DefaultLoadPlanner):
    def create_local_plan(self) -> LoadPlan:
        requests = []
        read_items: List[ReadItem]
        for fqn, obj in self.state_dict.items():  # pyre-ignore[16]
            md = self.metadata.state_dict_metadata[fqn]  # pyre-ignore[16]
            if isinstance(obj, DTensor):
                read_items = _create_read_items_for_dtensor(fqn, md, obj)
            elif (
                not isinstance(obj, DTensor)
                or self.is_coordinator  # pyre-ignore[16]
            ):
                read_items = _create_read_items(fqn, md, obj)
            requests += read_items

        return LoadPlan(requests)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        obj = self.state_dict[index.fqn]  # pyre-ignore[16]
        if isinstance(obj, DTensor):
            return obj.to_local()
        else:
            return super().lookup_tensor(index)


def get_box_for(
    tensor: DTensor, idx: Optional[int]
) -> Tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    assert device_mesh.ndim == 1, "Only 1D DeviceMeshes currently handled"

    size = tensor.to_local().size()
    offsets = tensor._spec.local_offsets

    return (torch.Size(offsets), size)


def _create_write_items_for_dtensor(fqn: str, tensor: DTensor) -> WriteItem:
    offsets, sizes = get_box_for(
        tensor, tensor.device_mesh.get_coordinate_on_dim(0)
    )
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(
                offsets=offsets,
                sizes=sizes,
            ),
            properties=TensorProperties.create_from_tensor(tensor.to_local()),
            size=tensor.size(),
        ),
    )


def _create_shard_from_dtensor(tensor: DTensor) -> Shard:
    offsets, sizes = get_box_for(
        tensor, tensor.device_mesh.get_coordinate_on_dim(0)
    )
    return Shard(
        tensor=tensor.to_local(),
        metadata=ShardMetadata(
            shard_offsets=list(offsets),
            shard_sizes=list(sizes),
            placement=f"rank:{dist.get_rank()}/{tensor.to_local().device}",
        ),
    )


def _create_read_items_for_dtensor(
    fqn: str, md: STORAGE_TYPES, obj: Any  # pyre-ignore[2]
) -> List[ReadItem]:
    if isinstance(obj, DTensor):
        local_shards = [_create_shard_from_dtensor(obj)]
    else:
        raise ValueError(
            f"Invalid checkpoint metadata for {fqn}, "
            + f"expected BytesStorageMetadata but found {type(md)}"
        )

    sharded_read_items = _create_sharded_read_items(fqn, md, local_shards)  # type: ignore # pyre-ignore[6]
    return sharded_read_items
