# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.distributed.rpc as rpc

from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

# WAR for SEV remediation https://github.com/pytorch/pytorch/commit/2337d4e5036a87f473decd2b1f6fe0439499902c
torch.fx.Tracer.proxy_buffer_attributes = True

if local_rank == 0:
    d_hid = 512
    bs = 503

    REPLICATE = os.environ.get('REPLICATE', '0') != '0'
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if REPLICATE else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {REPLICATE} -> {MULTI_USE_PARAM_CONFIG}')

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
            self.lin = torch.nn.Linear(d_hid, d_hid)
            self.register_buffer('buffer', torch.randn(bs + 100, d_hid))

        def forward(self, x):
            x = torch.mm(x, self.mm_param)
            skip_connection = x
            x = torch.relu(x)
            pipe_split()
            x = torch.mm(x, self.mm_param) + self.buffer[:x.shape[0]]
            x = self.lin(x)
            pipe_split()
            x = torch.relu(x)
            x = x + skip_connection
            x = torch.mm(x, self.mm_param2)
            x = self.lin(x)
            return {'out': x}

    ec = ExampleCode()
    ec(torch.randn(bs, d_hid))

    ec_pipe = Pipe.from_tracing(ec, MULTI_USE_PARAM_CONFIG)

    optimizer = torch.optim.SGD(ec_pipe.parameters(), 0.01)

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec = {}
    output_chunk_spec = {'out': TensorChunkSpec(0)}

    pipe_driver = PipelineDriverFillDrain(ec_pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size)

    input = torch.randn(bs, d_hid)

    # # Warm up and correctness runs
    out = pipe_driver.run((input,), {}, chunks=5, _debug_mask_minibatches = True)
    ref_out = ec_pipe(input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_allclose(out['out'], ref_out['out'])
        print(f'equivalence test passed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}')
        
    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        out = pipe_driver.run((input,), {}, chunks=5, _debug_mask_minibatches = False)
        ref_out = ec_pipe(input)
        print(f'profiling run completed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace(f'{os.path.splitext(os.path.basename(__file__))[0]}.json')

rpc.shutdown()
