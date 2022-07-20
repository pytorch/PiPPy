# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
import socket
from typing import Dict

import torch
import transformers.utils.fx as fx
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torchvision.models import ResNet, resnet101, resnet18
from torchvision.models.resnet import _resnet, BasicBlock
from transformers import AlbertModel, AlbertConfig

from pippy.IR import MultiUseParameterConfig, Pipe
from pippy.PipelineDriver import (
    PipelineDriverBase,
    PipelineDriverFillDrain,
    PipelineDriver1F1B,
    PipelineDriverInterleaved1F1B,
)
from pippy.auto_parallelization import AutoParallelConfig, dp_auto_parallel
from pippy.microbatch import TensorChunkSpec
from test_commons import tp_transports  # type: ignore

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get("VERBOSE", False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)


@torch.fx.wrap
def torch_ones_wrapper(*args, **kwargs):
    return torch.ones(*args, **kwargs)


class HFBertTracer(fx.HFTracer):
    def trace(self, root, concrete_args=None):
        graph = super().trace(root, concrete_args)
        for node in graph.nodes:
            if node.op == "call_function":
                if getattr(node.target, "_orig", None) == torch.ones:
                    node.target = torch_ones_wrapper
        return graph


torch.fx.Tracer.proxy_buffer_attributes = True


def run_master(args):
    bs = 1

    MULTI_USE_PARAM_CONFIG = (
        MultiUseParameterConfig.REPLICATE
        if args.replicate
        else MultiUseParameterConfig.TRANSMIT
    )
    print(f"REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}")
    print("Using schedule:", args.schedule)

    # resnet = resnet18(weights=None)
    resnet = _resnet(BasicBlock, [1, 1, 1, 1], weights=None, progress=False)
    resnet.eval()
    resnet_input = torch.randn((bs, 3, 16, 16))
    resnet(resnet_input)

    auto_parallel_config = AutoParallelConfig(
        n_compute_nodes=4, n_devices_per_node=4, n_microbatches=5
    )
    resnet_pipe = Pipe.from_tracing(
        resnet,
        MULTI_USE_PARAM_CONFIG,
        example_inputs=(resnet_input,),
        auto_parallel_strategy=dp_auto_parallel(auto_parallel_config),
    )
    print(resnet_pipe.split_gm)

    return

    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec: Dict = {}
    output_chunk_spec = {"out": TensorChunkSpec(0)}

    pipe_driver: PipelineDriverBase = schedules[args.schedule](
        resnet_pipe,
        args_chunk_spec,
        kwargs_chunk_spec,
        output_chunk_spec,
        args.world_size,
        _debug_mask_minibatches=True,
        _record_mem_dumps=bool(args.record_mem_dumps),
        checkpoint=bool(args.checkpoint),
    )

    # # Warm up and correctness runs
    pipe_driver.chunks = 5
    out = pipe_driver(resnet_input)
    ref_out = resnet_pipe(resnet_input)

    # run with different chunk size to exercise microbatch and scheduling components
    pipe_driver.chunks = 1
    pipe_driver(resnet_input)
    pipe_driver.chunks = 100
    pipe_driver(resnet_input)

    if CHECK_NUMERIC_EQUIVALENCE:
        torch.testing.assert_close(out["out"], ref_out["out"])
        print(
            f'equivalence test passed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}'
        )

    # # Profiling runs
    with torch.autograd.profiler_legacy.profile(enabled=PROFILING_ENABLED) as prof:
        pipe_driver.chunks = 5
        out = pipe_driver(resnet_input)
        ref_out = resnet_pipe(resnet_input)
        print(
            f'profiling run completed {torch.sum(out["out"])} ref {torch.sum(ref_out["out"])}'
        )
    if PROFILING_ENABLED:
        prof.export_chrome_trace(
            f"{os.path.splitext(os.path.basename(__file__))[0]}.json"
        )


def run_worker(rank, world_size, args):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=256, rpc_timeout=1800, _transports=tp_transports()
    )
    if args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
            for i in range(world_size):
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
        else:
            args.cuda = 0

    args.device = f"cuda:{dev_id}" if args.cuda else "cpu"
    print(
        f"rank = {rank} host/pid/device = "
        f"{socket.gethostname()}/{os.getpid()}/{args.device}"
    )

    rpc.init_rpc(
        f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options
    )
    if rank == 0:
        run_master(args)
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 1))
    )
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", 0)))
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29520")
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
    parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--record_mem_dumps", type=int, default=0, choices=[0, 1])
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    # Interleaved 1F1B uses less ranks than number of stages
    if args.schedule == "Interleaved1F1B":
        args.world_size = 2

    if args.rank == -1:
        mp.spawn(
            run_worker, args=(args.world_size, args), nprocs=args.world_size, join=True
        )
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
