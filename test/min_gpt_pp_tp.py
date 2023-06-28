# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

import pippy
import pippy.fx

import torch

import torch.distributed as dist
import torch.distributed.tensor.parallel as tp
from min_gpt_tracing import AdditionDataset  # type: ignore

from minGPT.mingpt.model import GPT, GPTConfig
from pippy.IR import annotate_split_points, PipeSplitWrapper
from torch.distributed._tensor import DeviceMesh

pippy.fx.Tracer.proxy_buffer_attributes = True

batch_size_per_chunk = 8

# The seed here has two purposes:
# - Ensure all TP ranks have same input
# - Ensure the model (ec) created are the same, as if it comes from a
# single, big model before partitioning
torch.manual_seed(0)

ndigit = 2
train_dataset = AdditionDataset(ndigit=ndigit, split="train")
test_dataset = AdditionDataset(ndigit=ndigit, split="test")

mconf = GPTConfig(
    train_dataset.vocab_size,
    train_dataset.block_size,
    n_layer=4,
    n_head=4,
    n_embd=128,
)

d_hid = 4


def run_all(args):
    # initialize a baby GPT model
    model = GPT(mconf)
    model.eval()
    model.to(args.device)

    # Specify split points
    sp_spec = {
        "blocks.0.mlp.3": PipeSplitWrapper.SplitPoint.END,
        "blocks.1.mlp.3": PipeSplitWrapper.SplitPoint.END,
        "blocks.2.mlp.3": PipeSplitWrapper.SplitPoint.END,
    }
    annotate_split_points(model, sp_spec)

    # Create input
    x = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    batch_size = args.chunks * batch_size_per_chunk
    device_type = args.device.type
    inp = x.repeat(batch_size, 1).to(args.device)

    # Create global DeviceMesh
    ranks = torch.arange(args.world_size)
    rank_mesh = ranks.reshape(args.pp_group_size, args.tp_group_size)
    pp_dim = 0
    tp_dim = 1
    dev_mesh = DeviceMesh(
        device_type,
        rank_mesh,
    )

    # Figure out my PP and TP rank
    pp_rank = args.rank // args.tp_group_size
    tp_rank = args.rank % args.tp_group_size
    print(
        f"Global rank {args.rank}, pp rank: {pp_rank}, tp rank: {tp_rank}, device: {args.device}"
    )

    # Get pp group
    # `tp_rank` can serve as pipeline id
    print(
        f"Rank {args.rank} Instantiating pipeline with ranks {dev_mesh.mesh[:, tp_rank]}"
    )
    pp_group = dev_mesh.get_dim_groups()[pp_dim]

    # Get stage module (on all pp ranks)
    stage = pippy.compile_stage(
        model,
        pp_rank,
        args.pp_group_size,
        args.chunks,
        args.device,
        pp_group,
        example_inputs=[inp],
        concrete_args={"targets": None},
    )

    # Tensor parallelize submodules
    print(f"Rank {args.rank} TP-lize submodule with {dev_mesh.mesh[pp_rank]}")
    tp.parallelize_module(
        stage.submod,
        dev_mesh,
        parallelize_plan={
            f"blocks_{pp_rank}_mlp_0": tp.ColwiseParallel(),
            f"blocks_{pp_rank}_mlp_2": tp.RowwiseParallel(),
        },
        tp_mesh_dim=tp_dim,
    )

    if pp_rank == 0:
        out = stage(None, inp)
    else:
        out = stage()

    dist.barrier()
    print(f"Rank {args.rank} completes")

    # Last rank checks result
    if pp_rank == args.pp_group_size - 1:
        ref_out = model(inp)[0]  # [0] is logits, [1] is loss (None)
        torch.testing.assert_close(out, ref_out)
        print(
            f"Pipeline {tp_rank} equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}"
        )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 8))
    )
    # ExampleCode has 4 stages
    parser.add_argument(
        "--pp_group_size",
        type=int,
        default=4,
    )
    # in row-major
    # TP ranks are contiguous rows of size `args.tp_group_size`
    # PP ranks are non-contiguous columns of size `args.pp_group_size`
    #
    # if tp_group_size = 4 and pp_group_size = 3
    #
    #   0 1 2  3
    #   4 5 6  7
    #   8 9 10 11
    #
    # TP ranks are [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    # PP ranks are [0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=4,
    )
    args = parser.parse_args(args)

    # Use world size to determine TP group size
    assert args.world_size % args.pp_group_size == 0
    args.tp_group_size = args.world_size // args.pp_group_size
    if args.rank == 0:
        print(
            f"Pipeline parallel size: {args.pp_group_size}\n"
            f"Tensor parallel size: {args.tp_group_size}"
        )

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
        # HACK: we need to pin device here because `DeviceMesh` currently does
        # an all_gather with device_type only, without device id
        # https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/device_mesh.py#L191-L192
        torch.cuda.set_device(args.device)
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run_all(args)


if __name__ == "__main__":
    main()
