import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (Any, Dict, ItemsView, Iterator, KeysView, List,
                    MutableMapping, Optional, Tuple, TypeVar, ValuesView)

import torch
import torch.fx as fx

# some pytorch low-level memory management constant the minimal allocate memory
# size (Byte)
PYTORCH_MIN_ALLOCATE: int = 2**20
# the minimal cache memory size (Byte)
PYTORCH_MIN_CACHE: int = 2**20
# default device for graph based profiling


KT = TypeVar("KT")
VT = TypeVar("VT")


# Bidirectional Dictionary to store the mapping of the forward and backward pass
# intermediate nodes
class BiDict(MutableMapping[KT, VT]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.data: Dict[KT, VT] = dict(*args, **kwargs)
        self.inverse: Dict[VT, List[KT]] = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __getitem__(self, key: KT) -> VT:
        return self.data.__getitem__(key)

    def __setitem__(self, key: KT, value: VT) -> None:
        if key in self:
            self.inverse[self[key]].remove(key)
        self.data.__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key: KT) -> None:
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        self.data.__delitem__(key)

    def __len__(self) -> int:
        return self.data.__len__()

    def __iter__(self) -> Iterator[KT]:
        return self.data.__iter__()

    def keys(self) -> KeysView[KT]:
        return self.data.keys()

    def values(self) -> ValuesView[VT]:
        return self.data.values()

    def items(self) -> ItemsView[KT, VT]:
        return self.data.items()


class GraphType(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class ProfileMode(str, Enum):
    r"""
    ProfileMode : The Graph Profiler provides three profiling
    modes,``default``, ``swap`` and ``mem_saver_swap``.
            default: Measure the per node run-time, marks the intermediate
                    nodes (activations saved from forward pass and needed in
                    backward pass), registers their last use in the forward
                    pass and first use in the backward pass, measures this
                    idle time and, marks the irst use of the model parameter
                    nodes in the forward pass.
            memory: All of the above plus active memory usage,
                    peak memory usage and intermediate (activation) memory.
            swap:   All the of the above plus profiles in a low memory
                    mode, pushing all of the activations to the CPU
                    memory during the forward pass and fetches them
                    back when they are needed in the backward pass.
                    It measures the time to swap each of the intermediate
                    tensors (activations) to CPU memory, back and forth.
                    Allows profiling graphs way larger than GPU memory.
    """
    default = "default"
    memory = "memory"
    swap = "swap"


class TensorStatus(Enum):
    cpu = auto()
    gpu = auto()
    deleted = auto()
    recomputed = auto()


def same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    return x.storage().data_ptr() == y.storage().data_ptr()


def get_tensor_stats(tensor: torch.Tensor) -> Tuple[Tuple[int, ...], int, int]:
    r"""
    Utility method that provides stats on the queried tensor. Args:
        tensor (torch.Tensor): Input tensor to get the stats for
    Returns:
        Tuple(size, numel, memory_size):
            size: the dimensions of ``tensor`` numel: number of elements in the
            ``tensor`` memory_size: the physical memeory occupied by the
            ``tensor`` in
                        bytes.
    """
    if tensor.is_sparse:
        raise NotImplementedError

    size = tuple(tensor.size()) if tensor.dim() > 0 else (1,)
    numel = tensor.numel()
    element_size = tensor.storage().element_size()
    fact_numel = tensor.storage().size()
    fact_memory_size = fact_numel * element_size
    # rounding up to pytorch's allocation granularity
    memory_size = (
        math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE)
        * PYTORCH_MIN_ALLOCATE
    )
    return (size, numel, memory_size)


@dataclass
class NodeInfo:
    r"""
    The base class to store the profiling and static graph analysis information
    for all the nodes in the graph.
    """
    # Stores the rank of the node in the order that is executed.
    rank: int = 0
    # Denotes if the node belongs to the forward/backward graph.
    gtype: Optional[GraphType] = None
    # The recorded run-time to the node in ms.
    run_time: float = 1.0
    cuda_time: float = 1.0
    cpu_time: float = 1.0
    # The cumulative run-time of the node from the first node in the graph.
    cumulative_run_time: float = 1.0
    # The peak memory usage of the node during execution in bytes.
    peak_mem: int = 0
    # The number of bytes active subsequent to the node's execution.
    active_mem: int = 0
    # Flags is True if the node lies in the peak memory interval(when active
    # memory exceeds device memory limit)
    in_peak_interval: bool = False
    # This is the peak memory consumption calculated by simulating the memory
    # usage for low memory mode.
    total_peak_mem: int = 0
    # Reference to the node that first uses self in the forward pass. Generally
    # populated for the parameter nodes of the model.
    first_forward_access: Optional[fx.Node] = None
    # Reference to the node that last uses self in the forward pass. Generally
    # populated for the intermediate (activation) nodes.
    last_forward_access: Optional[fx.Node] = None
    # Reference to the node that first uses self in the backward pass. Generally
    # populated for the intermediate (activation) nodes.
    first_back_access: Optional[fx.Node] = None
    # Reference to the node that last uses self in the backward pass. Generally
    # populated for the intermediate (activation) nodes.
    last_back_access: Optional[fx.Node] = None
    # Populated during scheduling algorithm (Future use)
    fw_nodes: List[fx.Node] = field(default_factory=list)
    fw_backup: List[fx.Node] = field(default_factory=list)
    intermediate_mem: float = 0.0
    to_offload: List[fx.Node] = field(default_factory=list)
    to_prefetch: List[fx.Node] = field(default_factory=list)
    to_recompute: List[fx.Node] = field(default_factory=list)
    to_delete: List[fx.Node] = field(default_factory=list)


class IntermediateNodeInfo(NodeInfo):
    r"""
    Derieved class to store the profiling and static graph analysis information
    for intermediate nodes (activations) in the graph.
    """
    # The idle time is calculated as [(last_backward_acess - swap_time) -
    # (last_forward_access + swap_time)].
    idle_time: float = 0.0
    # The time in ms required to swap tensor to and fro CPU memory.
    swap_time: float = 0.0
    # The dimension of the intermediate tensor.
    size: Tuple[int, ...] = (0,)
    # The physical memeory occupied by the tensor in bytes
    memory_size: int = 0
    # The number of elements in the tensor.
    numel: int = 0
    # The reference to the pinned memory CPU tensor.
    cpu_ref: Optional[torch.Tensor] = None
    # The current status of the tensor (CPU/GPU/Deleted)
    status: TensorStatus = TensorStatus.deleted

    # Attributes related to swap, populated during scheduling algorithm
    prefetch_trigger_start: Optional[fx.Node] = None
    prefetch_trigger_end: Optional[fx.Node] = None
    prefetch_begin: Optional[torch.cuda.Event] = None
    prefetch_end: Optional[torch.cuda.Event] = None
    offload_begin: Optional[torch.cuda.Event] = None
    active_forward_interval: List[fx.Node] = field(default_factory=list)
    active_backward_interval: List[fx.Node] = field(default_factory=list)

    # Attributes related to recomputation only, populated during recomp
    rcomp_sources: List[fx.Node] = field(default_factory=list)
    rcomp_primals: List[fx.Node] = field(default_factory=list)
    rcomp_extra: List[fx.Node] = field(default_factory=list)
    rcomp_graph_mod: Optional[fx.GraphModule] = None
    rcomp_executor: Optional[fx.Interpreter] = None

    exe_count: int = 0
    rcomp_time: float = 0.0
    exe_time: float = 0.0
    rcomp_mem: int = 0
    MSPS: float = 0.0
    is_recomp: bool = False

    def update_MSPS(self) -> None:
        # The metric currently being used in recomputation algorithm.
        self.MSPS = self.memory_size / self.exe_time


@dataclass
class ProfInfo:
    run_time: float
    cuda_time: float
    cpu_time: float
    cumulative_run_time: float
    peak_mem: int
    active_mem: int
    in_peak_interval: bool
    total_peak_mem: int
    idel_time: Optional[float] = None
    swap_time: Optional[float] = None
    size: Optional[int] = None
    memory_size: Optional[int] = None
    numel: Optional[int] = None
