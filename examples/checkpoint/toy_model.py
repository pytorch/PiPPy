# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

import torch
from torch.utils.data import Dataset, random_split
import torch.distributed as dist
import torch.optim as optim

from pippy.hf._SaveModule import save_checkpoint
from pippy.compile import compile_stage
from pippy.IR import pipe_split


d_hid = 512
chunk_size = 256

torch.manual_seed(0)


class RandomCustomDataset(Dataset):
    def __init__(self, chunks=1, size=100):  # TODO: reset size to 10000
        self.samples = [torch.randn(chunks * chunk_size, d_hid) for _ in range(size)]
        self.targets = [torch.randn(chunks * chunk_size, d_hid) for _ in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, x, target):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
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
        loss = self.mse_loss(x, target)
        return {"loss": loss}


def run_worker(args):
    ec = ExampleCode()
    ec.to(args.device)
    ec.train()

    ec_x = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)
    target = torch.randn(args.chunks * chunk_size, d_hid, device=args.device)

    ds = RandomCustomDataset(chunks=args.chunks)
    train_size = int(0.7*len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])
    datasets = {
        "train": train_ds,
        "test": test_ds,
    }

    stage = compile_stage(
        ec,
        args.rank,
        args.world_size,
        args.chunks,
        args.device,
        None,
        [ec_x, target],
    )

    # Create an optimizer for stage submodule's parameters
    optimizer = optim.SGD(stage.submod.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(args.epochs):  # change to no. epochs
        print(f"Epoch: {epoch + 1}")

        # save checkpoint
        if (epoch + 1) % args.checkpoint_epochs == 0:  # save ckpt after every `args.checkpoint_epochs` epochs
            print("RUuuuuuuuuuuunnnnnnnnnninngggggggggg")
            save_checkpoint(stage, checkpoint_dir=os.path.join("checkpoints", str(epoch + 1)), optimizer=optimizer)
            print("Doooooooooooonnnnnnnnnnnnnneeeeeeeeeee")

        for k, dataset in datasets.items():
            epoch_correct = 0
            epoch_all = 0
            for i, (x, y) in enumerate(dataset):
                x = x.to(args.device)
                y = y.to(args.device)
                if k == "train":
                    # Zero gradients
                    optimizer.zero_grad()
                    # Run
                    if args.rank == 0:
                        out = stage(x)
                    elif args.rank == args.world_size - 1:
                        out = stage(target)
                        out_tensor = out['loss']
                        preds = out_tensor.argmax(-1)
                        correct = (preds == y).sum()
                        epoch_all += len(y)
                        epoch_correct += correct.item()
                        # Take an optimization step
                        optimizer.step()
                    else:
                        stage()
                else:
                    stage.eval()
                    with torch.no_grad():
                        if args.rank == 0:
                            out = stage(x)
                        elif args.rank == args.world_size - 1:
                            out = stage(x)['loss']
                            preds = out.argmax(-1)
                            correct = (preds == y).sum()
                            epoch_all += len(y)
                            epoch_correct += correct.item()
                        else:
                            stage(x)
            # print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")

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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--chunks",
        type=int,
        default=4,
    )
    parser.add_argument("--checkpoint_epochs", type=int, default=1)
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
