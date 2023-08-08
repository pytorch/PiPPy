# Copyright (c) Meta Platforms, Inc. and affiliates
import os

import torch
import torch.distributed as dist

from model import GPT, GPTConfig
from pippy.compile import compile_stage

from pippy.IR import annotate_split_points, PipeSplitWrapper
from pippy.microbatch import sum_reducer, TensorChunkSpec

from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PairwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

from examples.selective2d.utils import *
from examples.selective2d.parallelize_utils import *


def even_cut(model, args, pp_size):
    """
    Evenly cut a model into pp_size stages
    """
    cut = {}
    cutpoint = args.n_layer // pp_size
    for i in range(args.n_layer):
        name = f"transformer.h.{i}"
        if i > 0 and i % cutpoint == 0:
            cut[name] = PipeSplitWrapper.SplitPoint.BEGINNING  # or END

    annotate_split_points(model, cut)


def after_ar_cut(model, args, pp_size):
    """
    Cut a model right after AllReduce happens
    """
    cut = {}
    cutpoint = args.n_layer // pp_size
    for i in range(args.n_layer):
        name = f"transformer.h.{i}"
        if i != args.n_layer - 1 and i % cutpoint == cutpoint - 1:
            cut[f"{name}.mlp.dropout"] = PipeSplitWrapper.SplitPoint.BEGINNING

    annotate_split_points(model, cut)


def tp_mlp(model, name, mesh, tp_dim=0, mlp="mlp"):
    layer = model.get_submodule(name)
    parallelize_module(
        layer, mesh, {mlp: PairwiseParallel()}, tp_mesh_dim=tp_dim
    )

    return model


def tp_attention(model, name, mesh, tp_dim=0, q="q", k="k", v="v", o="c_proj"):
    layer = model.get_submodule(name)
    parallelize_module(
        layer,
        mesh,
        {
            q: ColwiseParallel(),
            k: ColwiseParallel(),
            v: ColwiseParallel(),
            o: RowwiseParallel(),
        },
        tp_mesh_dim=tp_dim,
    )

    return model


def pp_and_tp_selective(
    model, mesh, args, tp_attn_layers=None, tp_mlp_layers=None, cut_fn=even_cut
):
    """
    Apply pipeline parallelism and tensor parallelism to a model.
    """

    pp_dim, tp_dim = 0, 1
    pp_rank, tp_rank = args.rank // args.tp_size, args.rank % args.tp_size
    pp_groups = mesh.get_dim_groups()[pp_dim]

    # TP
    # Apply TP to layers if layer_id is in tp_attn / tp_mlp
    tp_attn_layers = (
        list(range(args.n_layer)) if tp_attn_layers is None else tp_attn_layers
    )
    tp_mlp_layers = (
        list(range(args.n_layer)) if tp_mlp_layers is None else tp_mlp_layers
    )
    for i in range(args.n_layer):
        name = f"transformer.h.{i}"
        att = tp_attention(model, f"{name}.attn", mesh, tp_dim)
        mlp = tp_mlp(model, f"{name}", mesh, tp_dim)

    X, Y = get_rand_int(args)

    # PP
    cut_fn(model, args, args.pp_size * args.i_stage)
    num_stages = args.pp_size * args.i_stage

    if args.inference:
        stage = compile_stage(
            model,
            pp_rank,
            args.pp_size,
            args.n_chunks,
            args.device,
            pp_groups,
            example_inputs=[X, Y],
            num_stages=num_stages,
            schedule="TwoLevel",
        )
    else:
        output_chunk_spec = (TensorChunkSpec(0), sum_reducer)
        stage = compile_stage(
            model,
            pp_rank,
            args.pp_size,
            args.n_chunks,
            args.device,
            pp_groups,
            example_inputs=[X, Y],
            output_chunk_spec=output_chunk_spec,
            num_stages=num_stages,
            schedule="TwoLevel",
        )

    return model, stage


if __name__ == "__main__":
    _multi_gpu = int(os.environ.get("RANK", -1)) != -1  # verify distributed run
    assert (
        _multi_gpu
    ), "this config assumes distributed setup - multi-gpu not ready here."

    args = get_args_gpt2()

    device_type = (
        "cuda" if "cuda" in args.device else "cpu"
    )  # for later use in torch.autocast
    torch.cuda.set_device(args.device)

    dist.init_process_group(
        backend=args.backend, rank=args.rank, world_size=args.world_size
    )

    if args.master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=None,
        dropout=args.dropout,
    )  # start with model_args from command line

    # init a new model from scratch
    rank_print("Initializing a new model from scratch")

    twod_mesh = DeviceMesh(
        device_type=device_type,
        mesh=torch.arange(0, args.world_size).view(-1, args.tp_size),
    )

    model_args["vocab_size"] = args.vocab_size

    gptconf = GPTConfig(**model_args)
    model = GPT(twod_mesh, gptconf, args.device, args.pp_size)
    model.to(args.device)

    _current_model_params = model.get_num_params() / 1e6

    model, stage = pp_and_tp_selective(model, twod_mesh, args)

    if args.inference:
        iter_count, iter_time = pp_tp_inference(
            stage, twod_mesh, args, data_fn=get_rand_int
        )
    else:
        iter_count, iter_time = pp_tp_train(
            stage, twod_mesh, args, data_fn=get_rand_int
        )

    # display run stats
    rank_print(f"\nTraining completed.\n")

    gpu_type = torch.cuda.get_device_name(0)
    gpu_count = dist.get_world_size()
    rank_print(f"\n----- Performance Stats --------\n")
    rank_print(f"\nModel Size:  {_current_model_params:.2f}M")
    rank_print(f"Run completed with {gpu_count} gpus, of type {gpu_type}")
    iter_avg = round(iter_time / iter_count, 4)
    rank_print(
        f"Avg iter speed (in seconds): {iter_avg}, with {iter_count} iterations averaged.\n"
    )

    dist.destroy_process_group()
