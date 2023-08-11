# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

import torch


def get_args_mlp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=1024)

    # distributed
    parser.add_argument(
        "--backend", type=str, default="nccl"
    )  # 'nccl', 'gloo', etc.
    parser.add_argument("--rank", type=int, default=int(os.environ["RANK"]))
    parser.add_argument(
        "--world_size", type=int, default=int(os.environ["WORLD_SIZE"])
    )
    parser.add_argument(
        "--device", type=str, default=f"cuda:{os.environ['LOCAL_RANK']}"
    )

    parser.add_argument("--train_iters", type=int, default=10)
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--pp_size", type=int, default=2)
    parser.add_argument("--i_stage", type=int, default=1)
    parser.add_argument("--n_chunks", type=int, default=2)
    parser.add_argument("--nstreams", type=int, default=2)
    parser.add_argument("--warmup_iters", type=int, default=2)

    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--inner_cut", dest="inner_cut", action="store_true")
    parser.add_argument("--inference", dest="inference", action="store_true")

    args = parser.parse_args()

    return args


def get_args_gpt2():
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
    parser.add_argument("--nstreams", type=int, default=2)
    parser.add_argument("--warmup_iters", type=int, default=2)

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
    x = torch.rand(
        (args.batch_size, args.hidden_size),
        device=args.device,
    )
    y = torch.rand(
        (args.batch_size, args.hidden_size),
        device=args.device,
    )
    return x, y


def get_rand_int(args):
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
