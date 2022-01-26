from IR import Pipe, MultiUseParameterConfig, pipe_split
import torch
import torch.fx
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import operator
import logging
import threading
import copy

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

PROFILING_ENABLED = True

import torch.distributed.rpc as rpc



# logging.getLogger().setLevel(logging.INFO)



rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

if local_rank == 0:
    d_hid = 512
    bs = 503

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.lin = torch.nn.Linear(d_hid, d_hid)

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
    ec(torch.randn(bs, d_hid))

    ec_pipe = Pipe.from_tracing(ec, MultiUseParameterConfig.TRANSMIT)

    optimizer = torch.optim.SGD(ec_pipe.parameters(), 0.01)

    # TODO: refactor into PipeDriver
    # remote_stage_executor_rrefs : Dict[str, torch.distributed.rpc.RRef] = {}

    # for rank, (name, mod) in enumerate(ec_pipe.split_gm.named_children()):
    #     remote_stage_executor_rrefs[name] = (rank, rpc.remote(rank, PipeStageExecutor, (mod,)))

    # interp = RemoteInterpreter(remote_stage_executor_rrefs, ec_pipe.split_gm)

    input = torch.randn(bs, d_hid)

    check_numeric_equivalence = True

    # Warm up and correctness runs
    out = interp.run(input, chunks=5, _debug_mask_minibatches = True)
    ref_out = ec_pipe.split_gm(input)

    if check_numeric_equivalence:
        torch.testing.assert_allclose(out, ref_out)
        print(f'equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}')
        
    # Profiling runts
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        out = interp.run(input, chunks=5, _debug_mask_minibatches = False)
        ref_out = ec_pipe.split_gm(input)
        print(f'profiling run completed {torch.sum(ref_out)} ref {torch.sum(ref_out)}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace('pipe.csv')

rpc.shutdown()
