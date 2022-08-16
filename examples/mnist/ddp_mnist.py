# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

import torch
import torch.distributed
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.functional import cross_entropy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms  # type: ignore
from tqdm import tqdm  # type: ignore

from pippy.IR import PipeSplitWrapper, LossWrapper

USE_TQDM = bool(int(os.getenv('USE_TQDM', '1')))


def run_master(args):
    torch.manual_seed(42)

    print("Using device:", args.device)

    number_of_workers = 3
    chunks = number_of_workers
    batch_size = args.batch_size * chunks

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    valid_data = datasets.MNIST('./data', train=False, transform=transform)

    train_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=args.rank, shuffle=False,
                                       drop_last=False)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    class OutputLossWrapper(LossWrapper):
        def __init__(self, module, loss_fn):
            super().__init__(module, loss_fn)

        def forward(self, input, target):
            output = self.module(input)
            return output, self.loss_fn(output, target)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        PipeSplitWrapper(nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )),
        PipeSplitWrapper(nn.Linear(64, 10))
    )

    wrapper = OutputLossWrapper(model, cross_entropy)
    wrapper.to(args.device)
    wrapper = DDP(wrapper)

    optimizer = optim.Adam(wrapper.parameters())

    loaders = {
        "train": train_dataloader,
        "valid": valid_dataloader
    }

    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")
        for k, dataloader in loaders.items():
            epoch_correct = 0
            epoch_all = 0
            for i, (x_batch, y_batch) in enumerate(tqdm(dataloader) if USE_TQDM else dataloader):
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)
                if k == "train":
                    wrapper.train()
                    optimizer.zero_grad()
                    outp, loss = wrapper(x_batch, y_batch)
                else:
                    wrapper.eval()
                    with torch.no_grad():
                        outp, _ = wrapper(x_batch, y_batch)
                preds = outp.argmax(-1)
                correct = (preds == y_batch).sum()
                all = len(y_batch)
                epoch_correct += correct.item()
                epoch_all += all
                if k == "train":
                    loss.backward()
                    optimizer.step()
            print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    if args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
        else:
            args.cuda = 0
    args.device = f'cuda:{dev_id}' if args.cuda else 'cpu'

    # Init DDP process group
    backend = "nccl" if args.cuda else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    args.rank = rank
    run_master(args)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 2)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
