import argparse
import json
import os
import shutil
import unittest
from copy import deepcopy
from typing import List

import torch

import torch.distributed as dist
import torch.optim as optim
from pippy.compile import compile_stage

from pippy.hf._SaveModule import _get_binary_filename, save_checkpoint
from pippy.IR import pipe_split, TrivialLossWrapper
from pippy.LoadModule import load_checkpoint


DEFAULT_FILENAME = "pytorch_model.bin.index.json"
CKPT_DIR = "test_ckpts"
WEIGHT_MAP = set(
    [
        "module.mm_param0",
        "module.mm_param1",
        "module.mm_param2",
        "module.lin0.weight",
        "module.lin0.bias",
        "module.lin1.weight",
        "module.lin1.bias",
    ]
)
D_HID = 512
CHUNK_SIZE = 256

bs = 503


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.mm_param1 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.mm_param2 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.lin0 = torch.nn.Linear(D_HID, D_HID)
        self.lin1 = torch.nn.Linear(D_HID, D_HID)
        self.register_buffer("buffer", torch.randn(CHUNK_SIZE, D_HID))
        self.register_buffer("buffer2", torch.randn(CHUNK_SIZE, D_HID))

    def forward(self, x):
        x = torch.mm(x, self.mm_param0)
        skip_connection = x
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1) + self.buffer
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2) + self.buffer2
        pipe_split()
        x = self.lin1(x)
        x = torch.relu(x)
        return x


def run_worker(args: List[str | int]) -> None:
    ec = ExampleCode()

    ec_x = torch.randn(args.chunks * CHUNK_SIZE, D_HID, device=args.device)

    stage = compile_stage(ec,
                          args.rank,
                          args.world_size,
                          args.chunks,
                          args.device,
                          None,
                          [ec_x])


def main(args: List[str | int] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
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

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run_worker(args)


if __name__ == "__main__":
    main()


class LocalCheckpointTest(unittest.TestCase):
    def test_index_file(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
