# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import time

from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split
from pippy.PipelineDriver import PipelineDriverFillDrain
from pippy.microbatch import TensorChunkSpec

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

import os
local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))

rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

# WAR for SEV remediation https://github.com/pytorch/pytorch/commit/2337d4e5036a87f473decd2b1f6fe0439499902c
torch.fx.Tracer.proxy_buffer_attributes = True

@torch.fx.wrap
def sleep(x, t):
  time.sleep(t)
  return x

if local_rank == 0:
    d_hid = 100
    bs = 400

    REPLICATE = os.environ.get('REPLICATE', '0') != '0'
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if REPLICATE else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {REPLICATE} -> {MULTI_USE_PARAM_CONFIG}')

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d_hid, d_hid)
            self.l2 = nn.Linear(d_hid, d_hid)
            self.l3 = nn.Linear(d_hid, d_hid)
            self.l4 = nn.Linear(d_hid, d_hid)

        def forward(self, x):
            x = self.l1(x)
            sleep(x, 1)
            pipe_split()
            x = self.l2(x)
            sleep(x, 1)
            pipe_split()
            x = self.l3(x)
            sleep(x, 1)
            pipe_split()
            x = self.l4(x)
            sleep(x, 1)
            return {'out': x}

    ec = ExampleCode()
    ec(torch.randn(bs, d_hid))

    ec_pipe = Pipe.from_tracing(ec, MULTI_USE_PARAM_CONFIG)

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec = {}
    output_chunk_spec = {'out': TensorChunkSpec(0)}

    pipe_driver = PipelineDriverFillDrain(ec_pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec, world_size)

    input = torch.randn(bs, d_hid)

    def merge_jsons(world_size):
        with open("result.json", "w") as res:
            lines = []
            for i in range(world_size):
                with open(f"{i}.json", "r") as f:
                    lines.extend([l.rstrip() for l in f.readlines()])
                os.remove(f"{i}.json")
            res.write("[\n")
            res.write(",\n".join(lines))
            res.write("\n]\n")

    # # Warm up and correctness runs
    for i in range(4):
        out = pipe_driver.run((input,), {}, chunks=4, _debug_mask_minibatches = True)

    merge_jsons(len(pipe_driver.remote_stage_executor_rrefs))
    # ref_out = ec_pipe(input)

    # if CHECK_NUMERIC_EQUIVALENCE:
    #     torch.testing.assert_allclose(out['out'], ref_out['out'])
    #     print(f'equivalence test passed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}')
    #
    # # # Profiling runs
    # with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
    #     out = pipe_driver.run((input,), {}, chunks=4, _debug_mask_minibatches = False)
    #     ref_out = ec_pipe(input)
    #     print(f'profiling run completed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}')
    # if PROFILING_ENABLED:
    #     prof.export_chrome_trace('pipe.csv')

rpc.shutdown()
