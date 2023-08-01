# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch
import torch.distributed as dist

from pippy.IR import pipe_split
from pippy.compile import compile_stage

from torch.profiler import profile, ProfilerActivity

from torch.distributed._tensor import DeviceMesh
from pippy.microbatch import sum_reducer, TensorChunkSpec

class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super(MLPModule, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class ExampleCode(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, target):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        loss = self.mse_loss(x, target)
        return {"logits": x, "loss": loss}

class ExampleCodeRef(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, target):
        x1 = self.mlp0(x)
        pipe_split()
        x2 = self.mlp1(x1)
        pipe_split()
        x3 = self.mlp2(x2)
        pipe_split()
        x4 = self.mlp3(x3)
        loss = self.mse_loss(x4, target)
        return {"logits": x4, "loss": loss, "intm": [x1,x2,x3,x4]}


def get_args():
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("true", "t", "1"):
            return True
        elif v.lower() in ("false", "f", "0"):
            return False
        else:
            raise ArgumentTypeError("Boolean expected.")

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
        "--device", type=str, default=f"cuda:{os.environ['LOCAL_RANK']}"
    )
    parser.add_argument(
        "--master_process",
        type=str_to_bool,
        default=bool(os.environ["RANK"] == 0),
    )
    parser.add_argument(
        "--backend", type=str, default="nccl"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512
    )
    parser.add_argument(
        "--batch_size", type=int, default=12
    )
    parser.add_argument(
        "--seed", type=int, default=0
    )
    parser.add_argument(
        "--n_chunks", type=int, default=4
    )

    args = parser.parse_args()

    return args

def main():
    assert torch.cuda.is_available()

    args = get_args()

    device_type = (
        "cuda" if "cuda" in args.device else "cpu"
    )  # for later use in torch.autocast
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)

    dist.init_process_group(
        backend=args.backend, rank=args.rank, world_size=args.world_size
    )

    mesh = DeviceMesh(device_type, list(range(args.world_size)))
    pp_groups = mesh.get_dim_groups()[0]

    model = ExampleCode(args.hidden_size)

    example_input = torch.randn(args.batch_size, args.hidden_size, device=args.device)
    example_target = torch.randn(args.batch_size, args.hidden_size, device=args.device)

    model.to(args.device)
    ref_out = model(example_input, example_target)

    stage = compile_stage(
      model, 
      args.rank,
      args.world_size,
      args.n_chunks,
      args.device,
      None,
      example_inputs=[example_input, example_target],
    )

    with profile(
      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
      if args.rank == 0:
        out = stage(example_input)
      elif args.rank == args.world_size - 1:
        out = stage(example_target)
      else:
        out = stage()
      prof.step()

    prof.export_chrome_trace(f"local_test_twostream_rank{args.rank}.json")

    dist.barrier()
    if args.rank == args.world_size - 1:
      torch.testing.assert_close(out['logits'], ref_out['logits'])

if __name__ == '__main__':
    main()

class LocalTestTwostreamTest(unittest.TestCase):
    def test_forward(self):
        import random

        port = random.randint(29500, 30000)
        args = [
            "--master_port",
            str(port),
        ]
        main(args)
