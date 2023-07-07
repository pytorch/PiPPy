import argparse
import json
import os
import shutil
import unittest
from copy import deepcopy
from typing import List
import logging

import torch

import torch.distributed as dist
import torch.optim as optim
from pippy.compile import compile_stage

from pippy.hf._SaveModule import _get_binary_filename, save_checkpoint, _get_param_size, _atomic_write, _save_params
from pippy.IR import pipe_split, TrivialLossWrapper, Pipe, QualnameMapMixin
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


def _save_index_without_buffers(module: torch.nn.Module,
    ckpt_index_filename: str = DEFAULT_FILENAME,
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Saves index file describing location of only weights(ignoring buffers) in checkpoints.

    Args:
        pipe (Pipe): pipeline graph module with weights to save
        ckpt_index_filename (str, optional): name of index file. Defaults to "pytorch_model.bin.index.json".
        checkpoint_dir (str, optional): directory to save checkpoint to. Defaults to "checkpoints".
    """
    index_dict = {}
    total_size = 0

    weight_map: Dict[str, str] = {}
    for param_name, param in module.named_parameters():
        binary_filename = "pytorch_model-00001-of-00001.bin" # _get_binary_filename(0)

        if param_name not in weight_map:
            total_size += _get_param_size(param)

        weight_map[param_name] = binary_filename

    index_dict["metadata"] = {"total_size": total_size}
    index_dict["weight_map"] = weight_map

    # serialize json
    json_str = json.dumps(index_dict, indent=4)

    filepath = os.path.join(checkpoint_dir, ckpt_index_filename)

    # create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # write index file atomically to avoid partial/corrupted writes
    _atomic_write(json_str, filepath)

    logging.info(f"Saved index file to {filepath}")


def _save_checkpoint_without_buffers(module: torch.nn.Module, checkpoint_dir: str = "checkpoints"):
    if dist.get_rank() == 0:
        _save_index_without_buffers(module, checkpoint_dir=checkpoint_dir)

    # save module's state_dict directly to ckpt binary
    torch.save(
        {k:v for k, v in module.state_dict().items()},
        os.path.join(checkpoint_dir, "pytorch_model-00001-of-00001.bin"),
        # os.path.join(checkpoint_dir, _get_binary_filename(0)),
    )


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.mm_param1 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.mm_param2 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.lin0 = torch.nn.Linear(D_HID, D_HID)
        self.lin1 = torch.nn.Linear(D_HID, D_HID)
        self.register_buffer("buffer", torch.randn(CHUNK_SIZE, D_HID), persistent=False)
        self.register_buffer("buffer2", torch.randn(CHUNK_SIZE, D_HID), persistent=False)

    def forward(self, x):
        x = torch.mm(x, self.mm_param0)
        skip_connection = x
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param1) # + self.buffer
        x = self.lin0(x)
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2) + self.buffer2
        x = self.lin1(x)
        x = torch.relu(x)
        return x


def run_worker(args: List[str | int]) -> None:
    ec = ExampleCode()

    ec_x = torch.randn(args.chunks * CHUNK_SIZE, D_HID, device=args.device)
    target = torch.randn(args.chunks * CHUNK_SIZE, D_HID, device=args.device)

    _save_checkpoint_without_buffers(ec, checkpoint_dir=CKPT_DIR)

    module = load_checkpoint(ec, os.path.join(CKPT_DIR, DEFAULT_FILENAME), args.device)


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
        world_size=1, #args.world_size,
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
        with self.assertRaises(RuntimeError):
            main(args)
