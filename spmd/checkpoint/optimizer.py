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

from .nested_dict import unflatten_state_dict
from .utils import (element_wise_add, element_wise_sub)

import torch


def gen_rank_device(global_rank):
    if torch.cuda.is_available():
        return f"cuda:{global_rank % torch.cuda.device_count()}"
    return "cpu"

def create_colwise_spec(pg=None):
    if pg is None:
        placements = [
            f"rank:{idx}/{gen_rank_device(idx)}"
            for idx in range(dist.get_world_size())
        ]
    else:
        placements = [
            f"rank:{idx}/{gen_rank_device(dist.distributed_c10d.get_global_rank(pg, idx))}"
            for idx in range(pg.size())
        ]
    return ChunkShardingSpec(
        dim=0,
        placements=placements,
    )

def is_nested_tensor(val: Any) -> bool:
    if isinstance(val, ShardedTensor):
        if len(val.local_shards()) == 0:
            return False
        if isinstance(val.local_shards()[0].tensor, ShardedTensor):
            return True
        if isinstance(val.local_shards()[0].tensor, DT):
            raise ValueError("Cannot handle DT nested insided ST")
    # Safety valve for when this eventually happen
    elif isinstance(val, DT) and isinstance(val._local_tensor, (DT, ShardedTensor)):
        raise ValueError("Cannot handle nested DT")
    return False

def alloc_tensor(props: TensorProperties, size: torch.Size):
    return torch.empty(
        size=size,
        dtype=props.dtype,
        layout=props.layout,
        requires_grad=props.requires_grad,
        pin_memory=props.pin_memory,
        device=torch.cuda.current_device()
    )

def get_state_dict_2d_layout(state_dict: STATE_DICT_TYPE) -> Tuple[Dict[str, Tuple[Sequence[int], Sequence[int]]], dist.ProcessGroup]:
    """
    We have to load the right TP slice of the optimizer state.
    This is not easy since the per-tensor slicing can't be inferred from checkpoint metadata.
    We take advantage of the model state_dict producing a sliced ST to figure out what we need to load.
    This is pretty fragile and it might be easier for FSDP to compute this info for us.

    Returns a dictionary where keys are the same of the state_dict and the value is a tuple of
    (offset, size) for the current rank TP slice.

    N.B. The state_dict *MUST* come from FSDP.sharded_state_dict.
    """
    specs = {}
    dp_pg = None
    for key, value in state_dict.items():
        specs[key] = (None, value.size())
        if is_nested_tensor(value):
            assert len(value.local_shards()) == 1, "Cannot handle ST with multiple shards"
            shard = value.local_shards()[0]
            specs[key] = (shard.metadata.shard_offsets, shard.metadata.shard_sizes)
            dp_pg = shard.tensor._process_group

    return (specs, dp_pg, )

# TODO does it make sense to move this to UberLoadPlanner?
class ReaderWithOffset(DefaultLoadPlanner):
    def __init__(self, fqn_to_offset) -> None:
        super().__init__()
        # str ->tuple(offset, size)
        self.fqn_to_offset = fqn_to_offset
    
    def create_local_plan(self) -> LoadPlan:
        requests = []
        self.translation = {}
        for fqn, obj in self.state_dict.items():
            md = self.metadata.state_dict_metadata[fqn]
            if not isinstance(obj, ShardedTensor):
                requests += _create_read_items(fqn, md, obj)
                continue

            if fqn not in self.fqn_to_offset:
                requests += _create_read_items(fqn, md, obj)
                continue
            
            offset = self.fqn_to_offset[fqn]

            assert len(obj.local_shards()) == 1
            original_shard = obj.local_shards()[0]
            shard_md = copy.deepcopy(original_shard.metadata)
            shard_md.shard_offsets = element_wise_add(shard_md.shard_offsets, offset)
            local_shards = [Shard(original_shard.tensor, shard_md)]

            reqs = _create_sharded_read_items(fqn, md, local_shards)
            # The WriteItems will have a displaced MetadataIndex, fix it.
            # BTW, we should change _create_sharded_read_items to have more ergnomic API
            for wi in reqs:
                original_offset = element_wise_sub(wi.dest_index.offset, offset)
                original_index = dataclasses.replace(wi.dest_index, offset=torch.Size(original_offset))
                self.translation[wi.dest_index] = original_index

            requests += reqs
        return LoadPlan(requests)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        return super().lookup_tensor(self.translation.get(index, index))


def load_sharded_optimizer_state_dict(model_state_dict, optimizer_key, storage_reader):
    """
    This will load a state_dict to be used in conjuntion with FSDP sharded optimizer state.

    # Save
    model: torch.nn.Model
    optim_params = model.parameters()
    optim = torch.optim.SGD(optim_params, lr=0.01)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "optimizer": FSDP.sharded_optim_state_dict(model, optim, optim_params),
            "model": model.state_dict()
        }
        dist_cp.save_state_dict(
            state_dict=optim_state,
            storage_writer=dist_cp.FileSystemWriter("checkpoint"),
            planner=UberSavePlanner()
        )

    # Load
    with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model_tp.state_dict()
        checkpoint = {
            "model" = model_state_dict
        }
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=dist_cp.FileSystemReader(checkpoint_file),
            planner=UberLoadPlanner()
        )
        model.load_state_dict(checkpoint["model_state"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict,
            optimizer_key="optimizer",
            storage_reader=dist_cp.FileSystemReader("checkpoint"), 
        )

        flattened_osd = FSDP.flatten_sharded_optim_state_dict(
            optim_state["optimizer"], model, optim_input
        )

        optim.load_state_dict(flattened_osd)
    """
    metadata = storage_reader.read_metadata()

    layout_specs, dp_pg = get_state_dict_2d_layout(model_state_dict)

    if dp_pg is None:
        sharding_spec = ChunkShardingSpec(
            dim=0,
            placements=[f"rank:{i}/cuda:{i}" for i in range(dist.get_world_size())]
        )
    else:
        sharding_spec = create_colwise_spec(dp_pg)
    # Create a state_dict for optimizer state
    state_dict = {}
    """
    If we assume that the whole optimizer state is put under a single key, say 'optim', we'd be able to encode this as follows:

    get all keys with path prefix: ('optim')

    For all tensor types we use the 3rd component as the key into spec_key, IE:
    ('optim', 'state', 'net1.bias', 'exp_avg') -> 'net1.bias'

    """
    fqn_to_offset = {}
    for key, value in metadata.state_dict_metadata.items():
        key_path = metadata.planner_data[key]
        if key_path[0] != optimizer_key:
            continue

        if isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
            continue
        value: TensorStorageMetadata
        if value.size.numel() == 1:
            state_dict[key] = alloc_tensor(value.properties, value.size)
        elif dp_pg is not None:
            state_dict[key] = _shard_tensor(alloc_tensor(value.properties, value.size), sharding_spec)
        else:
            spec_key = key_path[2]
            alloc_size = layout_specs.get(spec_key, (None, value.size))[1]

            st_md = sharding_spec.build_metadata(alloc_size, value.properties)
            local_shards = []
            current_rank = dist.get_rank(dp_pg)
            for shard_md in st_md.shards_metadata:
                if shard_md.placement.rank() != current_rank:
                    continue
                local_shards.append(
                    Shard(
                        tensor=alloc_tensor(value.properties, shard_md.shard_sizes),
                        metadata=shard_md,
                    )
                )

            st = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards, st_md, process_group=dp_pg
            )

            if spec_key in layout_specs and layout_specs[spec_key][0] is not None:
                fqn_to_offset[key] = layout_specs[spec_key][0]

            state_dict[key] = st

    # Whether we unflatten before or after doesn't matter
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=ReaderWithOffset(fqn_to_offset) if dp_pg is not None else None
    )

    state_dict = unflatten_state_dict(state_dict, metadata.planner_data)

    return state_dict

# TODO add non-FSDP optimizer support

