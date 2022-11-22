import logging
import os
import sys
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Set, Union

import torch
from torch import fx

# Algorithms from paper: Chu, C. (1992). A branch-and-bound algorithm to
# minimize total tardiness with different release dates. Naval Research
# Logistics, 39(2), 265–283
# https://sci-hub.se/10.1002/1520-6750(199203)39:2%3C265::aid-nav3220390209%3E3.0.co;2-l

# Additional references
# Chu, C., and Portmann, M.C., “Minimisation de la Somme des Retards pour les
# Problkmes d’ordonnancement h Une Machine,” Rapport de Recherche No.
# 1023,INRIA, France (1989)
# https://hal.inria.fr/inria-00075535/document

# Chu, C., and Portmann, M.C., “Some New Efficient Methods to Solve the
# n/1/r_i/Sum(T_i), Scheduling Problem,” European Journal of Operational Research, 56
# (1991).
# https://sci-hub.se/https://www.sciencedirect.com/science/article/abs/pii/037722179290071G


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
