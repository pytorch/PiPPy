from dataclasses import dataclass
from enum import auto, Enum


class SchedulingPolicy(Enum):
    FCFS = auto()
    GREEDY_IPRTT = auto()
    GREEDY_NDPRTT = auto()
    BRANCH_AND_BOUND = auto()


@dataclass
class Process:
    p_id: int
    arrival_time: float
    burst_time: float
    deadline: float
