from IR import Pipe

from typing import Callable
from torch.optim import Optimizer

class Runtime:
    # TODO: actually run as a pipeline
    def __init__(self, model : Pipe, loss_code : Callable, optimizer : Optimizer):
        assert isinstance(model, Pipe)
        self.model = model
        self.loss_code = loss_code
        self.optimizer = optimizer

    def __call__(self, *args, **kwargs):
        self.optimizer.zero_grad()
        model_output = self.model(*args, **kwargs)
        if isinstance(model_output, tuple):
            loss_scalar = self.loss_code(*model_output)
        else:
            loss_scalar = self.loss_code(model_output)
        print(loss_scalar)
        loss_scalar.backward()
        self._sync_shared_params()
        self.optimizer.step()

    def _sync_shared_params(self):
        for shared_param_mapping in self.model.replicated_params:
            grad_objs = []
            for submod_qualname, param_qualname in shared_param_mapping.items():
                submod = self.model.split_gm.get_submodule(submod_qualname)
                param = submod.get_parameter(param_qualname)
                assert param.grad is not None
                grad_objs.append(param.grad)
            summed_grads = torch.sum(torch.stack(grad_objs), dim = 0)
            for submod_qualname, param_qualname in shared_param_mapping.items():
                submod = self.model.split_gm.get_submodule(submod_qualname)
                param = submod.get_parameter(param_qualname)
                param.grad = summed_grads


import torch
from IR import MultiUseParameterConfig, pipe_split

class ExampleCode(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mm_param = torch.nn.Parameter(torch.randn(512, 512))
    self.mm_param2 = torch.nn.Parameter(torch.randn(512, 512))
    self.lin = torch.nn.Linear(512, 512)

  def forward(self, x):
    x = torch.mm(x, self.mm_param)
    skip_connection = x
    x = torch.relu(x)
    pipe_split()
    x = torch.mm(x, self.mm_param)
    x = self.lin(x)
    pipe_split()
    x = torch.relu(x)
    x = x + skip_connection
    x = torch.mm(x, self.mm_param2)
    x = self.lin(x)
    return x

ec = ExampleCode()
ec(torch.randn(50, 512))

ec_pipe = Pipe.from_tracing(ec, MultiUseParameterConfig.TRANSMIT)

def loss_code(x):
    return torch.sum(x)

optimizer = torch.optim.SGD(ec_pipe.parameters(), 0.01)

rt = Runtime(ec_pipe, loss_code, optimizer)
rt(torch.randn(50, 512))

ec_pipe_replicated = Pipe.from_tracing(ec, MultiUseParameterConfig.REPLICATE)
rt = Runtime(ec_pipe_replicated, loss_code, optimizer)
rt(torch.randn(50, 512))
