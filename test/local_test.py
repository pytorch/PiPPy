from typing import List
import unittest
import argparse
import shutil
import json
import os

from pippy.LoadModule import load_checkpoint
from pippy.IR import pipe_split, TrivialLossWrapper
from pippy.hf._SaveModule import _save_index, _get_binary_filename
from pippy.compile import compile_stage
import torch.distributed as dist
import torch.optim as optim
import torch

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
BS = 503
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
        self.register_buffer("buffer", torch.randn(BS + 100, D_HID))

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
    optimizer = optim.SGD(stage.submod.parameters(), lr=1e-10, momentum=0.9)

    # for _ in range(2):
    # Zero gradients
    optimizer.zero_grad()

    # iteration 1
    if args.rank == 0:
        stage(ec_x)
    elif args.rank == args.world_size - 1:
        stage(target)
    else:
        stage()

    # Take an optimization step
    optimizer.step()

    dist.barrier()
    print(f"Rank {args.rank}")
    
    # iteration 1
    # print(f"iter1 -- rank:{dist.get_rank()},stage.submod.state_dict():{stage.submod.state_dict()}")
    # print(f"rank:{dist.get_rank()},stage.split_gm.state_dict():{stage.split_gm.state_dict()}")
    # from copy import deepcopy
    # ref_state_dict = deepcopy(stage.submod.state_dict())
    # torch.save(stage.submod.state_dict(), _get_binary_filename(args.rank))

    if args.rank == 0:
        _save_index(stage.pipe, checkpoint_dir=CKPT_DIR)
    from copy import deepcopy
    ref_state_dict = deepcopy(stage.submod.state_dict())
    remap_submod_sd = {stage.submod.remap_qualname(k) : v for k, v in stage.submod.named_parameters()}
    print(remap_submod_sd.keys())
    import os
    torch.save(
        remap_submod_sd, os.path.join('test_ckpts', _get_binary_filename(args.rank))
    )

    # iteration 2
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

    # print(f"iter2 -- rank:{dist.get_rank()},stage.submod.state_dict():{stage.submod.state_dict()}")

    # state_dict_to_load_from_iteration_1 = torch.load(_get_binary_filename(args.rank))
    # stage.submod.load_state_dict(state_dict_to_load_from_iteration_1)

    # see if this works with load_checkpoint from LoadModule.py out of the box. 
    # If not, what are the gaps?
    stage.submod = load_checkpoint(stage.submod, "test_ckpts/pytorch_model.bin.index.json")
    print(f"iter3 -- rank:{dist.get_rank()},stage.submod.state_dict():{stage.submod.state_dict()}")

    # save index file in rank 0
    if args.rank == 0:
        _save_index(stage, checkpoint_dir=CKPT_DIR)

        filepath = os.path.join(CKPT_DIR, DEFAULT_FILENAME)
        with open(filepath) as f:
            content = f.read()
            data = json.loads(content)

        # check file written on disk to given location
        assert os.path.exists(filepath)

        # check total_size correct
        size_calc = sum(param.numel() for param in ec.parameters()) * 4
        assert size_calc == data["metadata"]["total_size"]

        # check all params present
        assert len(data["weight_map"]) == 7
        for param in WEIGHT_MAP:
            assert param in data["weight_map"]

        # remove test directory
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


class LocalIndexMetadataTest(unittest.TestCase):
    def test_index_file(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)