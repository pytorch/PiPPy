# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import copy
import os
import unittest

import torch
import torch.distributed.rpc as rpc

import pippy.fx
from pippy import run_pippy
from pippy.IR import (
    MultiUseParameterConfig,
    Pipe,
    TrivialLossWrapper,
    pipe_split,
)
from pippy.PipelineDriver import (
    PipelineDriverFillDrain,
    PipelineDriver1F1B,
    PipelineDriverBase,
    PipelineDriverInterleaved1F1B,
)

# TODOs for implementing forward/backward/loss with schedules:
# * ability to switch between full-batch loss vs. per-microbatch loss. shen mentioned
# this might change numerics. So we should have the ability to compute loss over
# the whole minibatch rather than doing it for each micro-batch

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}


def get_grad_from_executor(executor, qualname):
    mod = executor.local_value().mod
    if isinstance(mod, torch.nn.parallel.DistributedDataParallel):
        return mod.module.get_parameter(qualname).grad
    else:
        return mod.get_parameter(qualname).grad


pippy.fx.Tracer.proxy_buffer_attributes = True


def run_master(pp_ranks, args):
    torch.manual_seed(42)

    d_hid = 50
    bs = 503
    CHUNKS = 5
    DEBUG_MASK_MINIBATCHES = True
    MULTI_USE_PARAM_CONFIG = (
        MultiUseParameterConfig.REPLICATE
        if args.replicate
        else MultiUseParameterConfig.TRANSMIT
    )
    print(f"REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}")

    print("Using schedule:", args.schedule)

    def rand_zeros_or_ones(shape):
        return torch.randint(0, 2, shape).float()

    class ZeroOneLinear(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.w = torch.nn.Parameter(rand_zeros_or_ones((in_dim, out_dim)))

        def forward(self, x):
            return x @ self.w

    class ExampleCode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_param = torch.nn.Parameter(
                rand_zeros_or_ones((d_hid, d_hid))
            )
            self.mm_param2 = torch.nn.Parameter(
                rand_zeros_or_ones((d_hid, d_hid))
            )
            self.lin = ZeroOneLinear(d_hid, d_hid)
            self.register_buffer(
                "buffer", 0.00001 * rand_zeros_or_ones((bs + 100, d_hid))
            )

        def forward(self, x):
            x = torch.mm(x, self.mm_param)
            skip_connection = x
            x = torch.relu(x)
            x = torch.mm(x, self.mm_param) + self.buffer[: x.shape[0]]
            x = self.lin(x)
            pipe_split()
            x = torch.relu(x)
            x = x + skip_connection
            x = torch.mm(x, self.mm_param2)
            x = self.lin(x)
            return x

    ec = ExampleCode()
    ec.to(args.device)
    ec(torch.randn(bs, d_hid, device=args.device))
    ec.train()

    # TODO: works with sum, need to define semantics for e.g. mean
    mse_loss = torch.nn.MSELoss(reduction="sum")
    wrapper = TrivialLossWrapper(ec, mse_loss)
    ec_pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG)
    if args.rank == 0:
        print(ec_pipe.split_gm)

    pipe_driver: PipelineDriverBase = schedules[args.schedule](
        ec_pipe,
        CHUNKS,
        args.pp_group_size,
        all_ranks=pp_ranks,
        _debug_mask_minibatches=DEBUG_MASK_MINIBATCHES,
        _record_mem_dumps=bool(args.record_mem_dumps),
        checkpoint=bool(args.checkpoint),
    )
    print(f"Rank {args.rank} Instantiated pipe with ranks {pp_ranks}")

    pipe_driver.init_data_parallel(dp_group_size=args.dp_group_size)

    torch.manual_seed(args.rank)
    input = torch.randn(bs, d_hid, device=args.device)
    target = torch.randn(bs, d_hid, device=args.device)

    # TODO: distributed optimizer
    out = pipe_driver(input, target)

    print(f"Rank {args.rank} got loss value {out}")

    all_grad_qualnames = {k: None for k, v in ec_pipe.named_parameters()}

    pipe_grads = {}

    for name in all_grad_qualnames:
        assert "split_gm." in name
        _, module_name, param_qualname = name.split(".", maxsplit=2)

        assert module_name in pipe_driver.remote_stage_executor_rrefs
        rank, module_rref = pipe_driver.remote_stage_executor_rrefs[module_name]
        grad_value = rpc.rpc_sync(
            module_rref.owner(),
            get_grad_from_executor,
            (module_rref, param_qualname),
        )
        pipe_grads[name] = copy.deepcopy(grad_value)

    # User driver group as the DDP reference group
    wrapper_ddp = torch.nn.parallel.DistributedDataParallel(
        wrapper, process_group=args.driver_group
    )

    optim = torch.optim.SGD(wrapper_ddp.parameters(), lr=0.05)
    optim.zero_grad()
    with torch.autograd.profiler.profile(enabled=args.rank == 0) as prof:
        wrapper_out = wrapper_ddp(input, target)
        wrapper_out.backward()
    if prof:
        prof.export_chrome_trace("ref.json")

    not_close_grads = []
    ref_grads = {}

    for name in all_grad_qualnames:
        remapped_qualname = ec_pipe.remap_qualname(name)
        param = wrapper_ddp.module.get_parameter(remapped_qualname)
        assert (
            name in pipe_grads
        ), f"{name} not in pipe_grads keys {pipe_grads.keys()}"
        ref_grads[name] = copy.deepcopy(param.grad)
        if not torch.allclose(pipe_grads[name], ref_grads[name]):
            not_close_grads.append(name)

    if len(not_close_grads):
        raise AssertionError(f"Gradients not close: {not_close_grads}")

    print("Gradient equivalence test passed")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
    # in row-major
    # DP ranks are contiguous rows of size `args.dp_group_size`
    # PP ranks are non-contiguous columns of size `args.pp_group_size`
    #
    # if dp_group_size = 4 and pp_group_size = 3
    #
    #   0 1 2  3
    #   4 5 6  7
    #   8 9 10 11
    #
    # DP ranks are [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    # PP ranks are [0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument(
        "-s",
        "--schedule",
        type=str,
        default=list(schedules.keys())[0],
        choices=schedules.keys(),
    )
    parser.add_argument(
        "--replicate", type=int, default=int(os.getenv("REPLICATE", "0"))
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument(
        "--record_mem_dumps", type=int, default=0, choices=[0, 1]
    )
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    args = parser.parse_args(args)

    # ExampleCode has two stages
    args.pp_group_size = 2
    assert args.world_size % args.pp_group_size == 0

    # Use world size to determine DDP size
    args.dp_group_size = args.world_size // args.pp_group_size
    print(f"Using data parallel group size: {args.dp_group_size}")

    run_pippy(run_master, args)


if __name__ == "__main__":
    main()


class LocalTestDDP(unittest.TestCase):
    def test_ddp(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--cuda",
            os.getenv("USE_CUDA", "0"),
            "--master_port",
            str(port),
        ]
        main(args)
