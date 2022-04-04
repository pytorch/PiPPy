# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import socket
import time

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch.autograd import Function

from pippy.IR import MultiUseParameterConfig, Pipe, pipe_split
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B
from pippy.microbatch import TensorChunkSpec, CustomReducer
from pippy.profiler import merge_jsons

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
}

# WAR for SEV remediation https://github.com/pytorch/pytorch/commit/2337d4e5036a87f473decd2b1f6fe0439499902c
torch.fx.Tracer.proxy_buffer_attributes = True


@torch.fx.wrap
def sleep(x):
    time.sleep(1)
    return x


class SlowMSELoss(nn.MSELoss):
    def forward(self, input, target):
        return super().forward(sleep(input), target)


# Inherit from Function
class MyLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        print("my forward")
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        print("my backward")
        grad_output = sleep(grad_output)
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class MyLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(MyLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MyLinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

def run_main(args):
    d_hid = 100
    bs = 400

    print(f'World size: {args.world_size}')
    REPLICATE = os.environ.get('REPLICATE', '0') != '0'
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if REPLICATE else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {REPLICATE} -> {MULTI_USE_PARAM_CONFIG}')

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = MyLinear(d_hid, d_hid)
            self.l2 = MyLinear(d_hid, d_hid)
            self.l3 = MyLinear(d_hid, d_hid)
            self.l4 = MyLinear(d_hid, d_hid)

        def forward(self, x):
            x = self.l1(x)
            sleep(x)
            pipe_split()
            x = self.l2(x)
            sleep(x)
            pipe_split()
            x = self.l3(x)
            sleep(x)
            pipe_split()
            x = self.l4(x)
            sleep(x)
            return x

    ec = ExampleCode()

    ec(torch.randn(bs, d_hid))

    mse_loss = SlowMSELoss(reduction='sum')

    ec_pipe = Pipe.from_tracing(ec, MULTI_USE_PARAM_CONFIG, loss_fn=mse_loss)

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}
    output_chunk_spec = CustomReducer(torch.tensor(0.0), lambda a, b: a + b)

    pipe_driver = PipelineDriverFillDrain(ec_pipe, args_chunk_spec, kwargs_chunk_spec, output_chunk_spec,
                                          args.world_size)

    input = torch.randn(bs, d_hid)
    target = torch.randn(bs, d_hid)

    # # Warm up and correctness runs
    for i in range(4):
        out = pipe_driver.run((input, target), {}, chunks=4, _debug_mask_minibatches=True)


def run_worker(rank, world_size, args):
    print(f"rank = {rank} host/pid = {socket.gethostname()}/{os.getpid()}")
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    if rank == 0:
        run_main(args)
        merge_jsons(world_size, "result.json")
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    args = parser.parse_args()

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
