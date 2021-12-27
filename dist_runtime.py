from IR import Pipe, MultiUseParameterConfig, pipe_split
import torch
import torch.fx
from typing import Dict
import operator

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

import torch.distributed.rpc as rpc

def move(module):
    return module

def to_here(a):
    if isinstance(a, torch._C._distributed_rpc.PyRRef):
        return a.to_here()
    else:
        return a

def invoke(mod_rref, args, kwargs):
    args = torch.fx.node.map_aggregate(args, to_here)
    kwargs = torch.fx.node.map_aggregate(kwargs, to_here)
    out = mod_rref.to_here()(*args, **kwargs)
    print(f'invoked target {mod_rref} on rank {local_rank}')
    return out

def tuple_idx(val_rref, idx):
    return val_rref.to_here()[idx]

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

if local_rank == 0:
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

    remote_module_rrefs : Dict[str, torch.distributed.rpc.RRef] = {}

    for rank, (name, mod) in enumerate(ec_pipe.split_gm.named_children()):
        remote_module_rrefs[name] = (rank, rpc.remote(rank, move, (mod,)))

    # Interpret top-level graph and issue remote calls

    class RemoteInterpreter(torch.fx.Interpreter):
        def __init__(self, remote_module_rrefs, module, garbage_collect_values = True):
            super().__init__(module, garbage_collect_values)
            self.remote_module_rrefs = remote_module_rrefs

        def call_module(self, target, args, kwargs):
            assert isinstance(target, str)

            if target in self.remote_module_rrefs:
                rank, mod_rref = self.remote_module_rrefs[target]
                return rpc.remote(rank, invoke, (mod_rref, args, kwargs))
            else:
                return super().call_module(target, args, kwargs)

        def call_function(self, target, args, kwargs):
            if target is operator.getitem and isinstance(args[0], torch._C._distributed_rpc.PyRRef):
                return rpc.remote(args[0].owner().id, tuple_idx, args)
            return super().call_function(target, args, kwargs)

    interp = RemoteInterpreter(remote_module_rrefs, ec_pipe.split_gm)

    input = torch.randn(50, 512)

    out = interp.run(input)

    ref_out = ec_pipe.split_gm(input)

    torch.testing.assert_allclose(out.to_here(), ref_out)

rpc.shutdown()