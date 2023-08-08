"""
This training script updates NanoGPT to run with either TP, PP, or TP+PP (2D).
Usage:
gpurun4 torchrun --nproc-per-node 4 2d_train.py 
"""

import argparse
import os
import time

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

from torch.profiler import profile, ProfilerActivity


def get_args():
    # default config values designed to train a gpt2 (124M) on OpenWebText

    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("true", "t", "1"):
            return True
        elif v.lower() in ("false", "f", "0"):
            return False
        else:
            raise ArgumentTypeError("Boolean expected.")

    # I/O
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=2)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument(
        "--eval_only", type=str_to_bool, default=False
    )  # if True, script exits right after the first eval
    parser.add_argument(
        "--always_save_checkpoint", type=str_to_bool, default=True
    )  # if True, always save a checkpoint after each eval
    parser.add_argument(
        "--init_from", type=str, default="scratch"
    )  # 'scratch', 'resume', 'gpt2*'
    parser.add_argument("--train_iters", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1337)

    # data
    parser.add_argument(
        "--dataset", type=str, default="shakespeare_char"
    )  # "openwebtext"
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )  # used to simulate larger batch sizes
    parser.add_argument(
        "--batch_size", type=int, default=12
    )  # if gradient_accumulation_steps > 1, this is the micro-batch size
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=50304)

    # model
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument(
        "--dropout", type=float, default=0.0
    )  # for pretraining 0 is good, for finetuning try 0.1+
    parser.add_argument("--bias", type=str_to_bool, default=False)

    # adamw optimizer
    parser.add_argument(
        "--learning_rate", type=float, default=4e-4
    )  # max learning rate
    parser.add_argument(
        "--max_iters", type=int, default=600000
    )  # total number of training iterations
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument(
        "--grad_clip", type=float, default=1.0
    )  # clip gradients at this value, or disable if == 0.0
    parser.add_argument(
        "--decay_lr", type=str_to_bool, default=True
    )  # whether to decay the learning rate
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--lr_decay_iters", type=int, default=600000)
    parser.add_argument(
        "--min_lr", type=float, default=6e-5
    )  # minimum learning rate

    # distributed
    parser.add_argument(
        "--backend", type=str, default="nccl"
    )  # 'nccl', 'gloo', etc.
    parser.add_argument(
        "--compile", type=str_to_bool, default=False
    )  # use PyTorch 2.0 to compile the model to be faster
    parser.add_argument("--rank", type=int, default=int(os.environ["RANK"]))
    parser.add_argument(
        "--local_rank", type=int, default=int(os.environ["LOCAL_RANK"])
    )
    parser.add_argument(
        "--world_size", type=int, default=int(os.environ["WORLD_SIZE"])
    )
    parser.add_argument(
        "--device", type=str, default=f"cuda:{os.environ['LOCAL_RANK']}"
    )
    parser.add_argument(
        "--master_process",
        type=str_to_bool,
        default=bool(os.environ["RANK"] == 0),
    )
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--i_stage", type=int, default=1)
    parser.add_argument("--n_chunks", type=int, default=2)

    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--inner_cut", dest="inner_cut", action="store_true")
    parser.add_argument("--inference", dest="inference", action="store_true")

    args = parser.parse_args()

    return args


def rank_print(x):
    _rank = os.getenv("RANK")
    if _rank == "0":
        print(x)


def get_rand(args):
    x = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.block_size),
        device=args.device,
    )
    y = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.block_size),
        device=args.device,
    )
    return x, y


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


def tp_mlp(model, name, mesh, tp_dim=0, mlp="mlp"):
    layer = model.get_submodule(name)
    parallelize_module(
        layer, mesh, {mlp: PairwiseParallel()}, tp_mesh_dim=tp_dim
    )

    return model


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

    X, Y = get_rand(args)

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


def pp_tp_train(stage, mesh, args):
    pp_dim, tp_dim = 0, 1
    pp_rank, tp_rank = args.rank // args.tp_size, args.rank % args.tp_size
    pp_groups = mesh.get_dim_groups()[pp_dim]

    train_iters = 10 if args.debug else args.train_iters
    optimizer = torch.optim.AdamW(
        stage.submod.parameters(), lr=args.learning_rate
    )
    local_iter_num = 0
    iter_time = 0.0
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=5, wait=0, warmup=4, active=1, repeat=1
        ),
    ) as prof:
        while local_iter_num < train_iters:
            optimizer.zero_grad()
            t0 = time.perf_counter()
            X, Y = get_rand(args)
            if pp_rank == 0:
                out = stage(X)
            elif pp_rank == args.pp_size - 1:
                out = stage(Y)
            else:
                out = stage()
            optimizer.step()
            t1 = time.perf_counter()
            dt = t1 - t0
            local_iter_num += 1
            iter_time += dt
            prof.step()

    prof.export_chrome_trace(f"trace_rank{args.rank}.json")

    return local_iter_num, iter_time


def pp_tp_inference(stage, mesh, args):
    pp_dim, tp_dim = 0, 1
    pp_rank, tp_rank = args.rank // args.tp_size, args.rank % args.tp_size
    pp_groups = mesh.get_dim_groups()[pp_dim]

    train_iters = 10 if args.debug else args.train_iters
    local_iter_num = 0
    iter_time = 0.0
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            skip_first=5, wait=0, warmup=4, active=1, repeat=1
        ),
    ) as prof:
        while local_iter_num < train_iters:
            t0 = time.perf_counter()
            X, Y = get_rand(args)
            if pp_rank == 0:
                out = stage(X)
            elif pp_rank == args.pp_size - 1:
                out = stage(Y)
            else:
                out = stage()
            t1 = time.perf_counter()
            dt = t1 - t0
            local_iter_num += 1
            iter_time += dt
            prof.step()
            dist.barrier()

    prof.export_chrome_trace(f"trace_rank{args.rank}.json")

    return local_iter_num, iter_time


if __name__ == "__main__":
    _multi_gpu = int(os.environ.get("RANK", -1)) != -1  # verify distributed run
    assert (
        _multi_gpu
    ), "this config assumes distributed setup - multi-gpu not ready here."

    args = get_args()

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

    oned_mesh = DeviceMesh(device_type, list(range(args.world_size)))
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
        iter_count, iter_time = pp_tp_inference(stage, twod_mesh, args)
    else:
        iter_count, iter_time = pp_tp_train(stage, twod_mesh, args)

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
