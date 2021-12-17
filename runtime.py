from IR import Pipe

from typing import Callable
from torch.optim import Optimizer

class Runtime:
    def __init__(self, model : Pipe, loss_code : Callable, optimizer : Optimizer):
        pass

    def __call__(self, *args, **kwargs):
        pass