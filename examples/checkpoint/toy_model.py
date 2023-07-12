# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.distributed as dist
import torch.optim as optim

from pippy.compile import compile_stage
from pippy.IR import pipe_split, TrivialLossWrapper


d_hid = 512
chunk_size = 256

torch.manual_seed(0)


class RandomCustomDataset(Dataset):
    def __init__(self, size=10000):
        self.samples = [torch.randn(d_hid, d_hid) for _ in range(size)]
        self.targets = [torch.randn(d_hid, d_hid) for _ in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = torch.mm(x, self.mm_param0)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin1(x)
        x = torch.relu(x)

        return x


def run_worker(args):
    ec = ExampleCode()
    loss_fn = torch.nn.MSELoss(reduction="sum")
    ec_with_loss = TrivialLossWrapper(ec, loss_fn)
    ec_with_loss.to(args.device)

    ec_x = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)
    target = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)

    ds = RandomCustomDataset()
    train_size = int(0.7*len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])
    train_dl, test_dl = DataLoader(train_ds), DataLoader(test_ds)
    loaders = {
        "train": train_dl,
        "test": test_dl,
    }

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

    for epoch in range(2):
        print(f"Epoch: {epoch + 1}")

        for k, loader in loaders.items():
            epoch_correct = 0
            epoch_all = 0

            for i, (x, y) in enumerate(loader):
                x = x.to(args.device)
                y = y.to(args.device)

                if k == "train":
                    # Zero gradients
                    optimizer.zero_grad()

                    # Run
                    if args.rank == 0:
                        stage(ec_x)
                    elif args.rank == args.world_size - 1:
                        out = stage(target)

                        print("###################################")
                        print(out)
                        print("###################################")

                    else:
                        stage()

                    # Take an optimization step
                    optimizer.step()

    dist.barrier()
    print(f"Rank {args.rank} completes")


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

## setup model, optimizer, lr-scheduler
