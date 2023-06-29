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

from pippy.hf._SaveModule import save_checkpoint, _get_binary_filename
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


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.mm_param1 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.mm_param2 = torch.nn.Parameter(torch.randn(D_HID, D_HID))
        self.lin0 = torch.nn.Linear(D_HID, D_HID)
        self.lin1 = torch.nn.Linear(D_HID, D_HID)

    def forward(self, x):
        x = torch.mm(x, self.mm_param0)
        skip_connection = x
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin1(x)
        x = torch.relu(x)
        return x


def run_worker(args: List[str | int]) -> None:
    ec = ExampleCode()
    loss_fn = torch.nn.MSELoss(reduction="sum")
    ec_with_loss = TrivialLossWrapper(ec, loss_fn)
    ec_with_loss.to(args.device)

    ec_x = torch.randn(args.chunks * CHUNK_SIZE, D_HID, device=args.device)
    target = torch.randn(args.chunks * CHUNK_SIZE, D_HID, device=args.device)

    stage = compile_stage(
        ec_with_loss,
        args.rank,
        args.world_size,
        args.chunks,
        args.device,
        None,
        [ec_x, target],
    )

    # Create an optimizer for stage submodule's parameters
    optimizer = optim.SGD(stage.submod.parameters(), lr=1e-3, momentum=0.9)

    # first run
    # Zero gradients
    optimizer.zero_grad()

    # Run
    if args.rank == 0:
        stage(ec_x)
    elif args.rank == args.world_size - 1:
        stage(target)
    else:
        stage()

    # Take an optimization step
    optimizer.step()
    submod_ref = deepcopy(stage.submod.state_dict())
    optim_ref = deepcopy(optimizer.state_dict())
    save_checkpoint(stage, CKPT_DIR, optimizer)

    # save index file in rank 0
    if args.rank == 0:
        filepath = os.path.join(CKPT_DIR, DEFAULT_FILENAME)
        with open(filepath) as f:
            content = f.read()
            data = json.loads(content)

        # check file written on disk to given location
        assert os.path.exists(filepath)

        # check total_size is correct
        size_calc = sum(param.numel() for param in ec.parameters()) * 4
        assert size_calc == data["metadata"]["total_size"]

        # check all params present
        assert len(data["weight_map"]) == 7
        for param in WEIGHT_MAP:
            assert param in data["weight_map"]

    # second run
    # Zero gradients
    optimizer.zero_grad()

    # Run
    if args.rank == 0:
        stage(ec_x)
    elif args.rank == args.world_size - 1:
        stage(target)
    else:
        stage()

    # Take an optimization step
    optimizer.step()

    # load ckpt
    mod = load_checkpoint(
        stage.submod,
        os.path.join(CKPT_DIR, "pytorch_model.bin.index.json"),
        args.device,
    )
    # load optim
    optimizer.load_state_dict(torch.load(os.path.join(CKPT_DIR, _get_binary_filename(dist.get_rank(), is_optim=True))))

    torch.testing.assert_close(mod.state_dict(), submod_ref)
    torch.testing.assert_close(optimizer.state_dict(), optim_ref)

    dist.barrier()
    print(f"Rank {args.rank} completes")

    # remove test ckpt directory in last rank
    if args.rank == args.world_size - 1:
        shutil.rmtree(CKPT_DIR)


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
