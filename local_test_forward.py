import torch
import torch.distributed.rpc as rpc

from IR import MultiUseParameterConfig, Pipe, pipe_split
from PipelineDriver import PipelineDriver

# LOG 1/18: Specifying schedule data dependencies via explicit dependencies is tricky because
#           it constrains the topological ordering in which the execution schedule can be
#           constructed. Instead, we could specify things like 1F1B scheduling by modeling
#           the resource constraint (e.g. Registers in OneFlow -- analogous to a semaphore)
#           and making the system block on this resource. zdevito pointed out that in this
#           case, parallel jobs may deadlock, as they can acquire resources in an arbitrary
#           order. This could be solved by specifying that acquiring this resource is an
#           external side effect and serializing all stages with external side effects
#           in the scheduling system.
# LOG 1/20: TODOs for implementing forward/backward/loss with schedules:
#           * ability to specify loss computation. Probably going to start with explicit callback
#             rather than a more complicated tracing system
#           * ability to switch between full-batch loss vs. per-microbatch loss. shen mentioned
#             this might change numerics. So we should have the ability to compute loss over
#             the whole minibatch rather than doing it for each micro-batch
#           * ability to schedule backwards
#
#           Plan of action:
#           * design representation in frontend/IR for forward/backward loss
#             (full mini-batch loss)
#           * Implement runtime for fill/drain pipelining (i.e. GPipe)
#           * Extend loss computation to be per-microbatch
#           * Implement 1F1B schedule

PROFILING_ENABLED = True

import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

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

    pipe_driver = PipelineDriver(ec_pipe, world_size)

    input = torch.randn(bs, d_hid)

    check_numeric_equivalence = True

    # # Warm up and correctness runs
    out = pipe_driver.run(input, chunks=5, _debug_mask_minibatches = True)
    ref_out = ec_pipe.split_gm(input)

    if check_numeric_equivalence:
        torch.testing.assert_allclose(out, ref_out)
        print(f'equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}')
        
    # # Profiling runts
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        out = pipe_driver.run(input, chunks=5, _debug_mask_minibatches = False)
        ref_out = ec_pipe.split_gm(input)
        print(f'profiling run completed {torch.sum(ref_out)} ref {torch.sum(ref_out)}')
    if PROFILING_ENABLED:
        prof.export_chrome_trace('pipe.csv')

rpc.shutdown()
