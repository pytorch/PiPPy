# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

from pippy.compile import compile_stage
from pippy.IR import pipe_split


d_hid = 512
chunk_size = 256

torch.manual_seed(0)


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = x + y
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param)
        x = self.lin(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin(x)
        x = torch.relu(x)
        return x


import json
from pippy.PipelineStage import PipelineStage


PYTORCH_BINARY_FILE_PREFIX = "pytorch_model"
JSON_INDEX_FILE_NAME = "pytorch_model.bin.index.json"


def get_binary_file_name(cur_idx: int):
    return f"{PYTORCH_BINARY_FILE_PREFIX}-0000{cur_idx}-of-0000{dist.get_world_size()}.bin"


def get_model_size(total_param:int, param_type:int) -> int:
    return total_param * 2

def generate_json_file(
    stage: PipelineStage,
    ckpt_dir: str = None,
) -> None:
    assert ckpt_dir is not None, "Please provide a checkpoint directory."
    ckpt_path = Path(ckpt_dir)

    weight_map = {}
    # num_sub_mod = len(stage.split_gm.named_children())
    # print(f"num_sub_mod:{num_sub_mod}")
    total_num_param = 0
    
    for idx, named_children_tuple in enumerate(stage.split_gm.named_children()):
        name, stage_mod = named_children_tuple
        for pippy_name, param in stage_mod.named_parameters():
            original_name = stage_mod.remap_qualname(pippy_name)
            print(f"new_name:{pippy_name}, old_name:{original_name}")
            weight_map[original_name] = get_binary_file_name(idx)
            if param.requires_grad:
                total_num_param += param.numel()
                param_type = param.dtype
                print(f'{param_type}, {type(param_type)}')

    json_map = {}
    json_map["meta_data"] = get_model_size(total_param=total_num_param, param_type=param_type)
    json_map["weight_map"] = weight_map
    with open(os.path.join(ckpt_path, "temp.json"), "w") as f:
        json.dump(json_map, f)
        os.fsync(f.fileno())

    (ckpt_path / "temp.json").rename(ckpt_path / f"{JSON_INDEX_FILE_NAME}")


def run_worker(args):
    ec = ExampleCode()
    ec.to(args.device)

    ec_x = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)
    ec_y = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)

    stage = compile_stage(
        ec,
        args.rank,
        args.world_size,
        args.chunks,
        args.device,
        None,
        [ec_x, ec_y],
    )

    # for name, param in stage.submod.named_parameters():
    #     print(f"{name}:{stage.submod.remap_qualname(name)}")
    if dist.get_rank() == 0:
        generate_json_file(stage, "pippy_checkpoint")

    # Run
    if args.rank == 0:
        out = stage(ec_x, ec_y)
    elif args.rank == args.world_size - 1:
        out = stage()
    else:
        stage()

    dist.barrier()
    print(f"Rank {args.rank} completes")

    # Last rank checks result
    if args.rank == args.world_size - 1:
        ref_out = ec(ec_x, ec_y)
        torch.testing.assert_close(out, ref_out)
        print(
            f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}"
        )


def main(args=None):
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

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run_worker(args)


if __name__ == "__main__":
    main()


class LocalTestC10DTest(unittest.TestCase):
    def test_c10d(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
