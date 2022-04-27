# Copyright (c) Meta Platforms, Inc. and affiliates
import os

import torch
from torch import nn, optim
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms  # type: ignore
from tqdm import tqdm  # type: ignore

from pippy.IR import PipeSplitWrapper, LossWrapper

USE_TQDM = os.getenv('USE_TQDM', True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    number_of_workers = 6
    all_worker_ranks = list(range(1, 1 + number_of_workers))  # exclude master rank = 0
    chunks = len(all_worker_ranks)
    batch_size = 10 * chunks

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    valid_data = datasets.MNIST('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)


    class OutputLossWrapper(LossWrapper):
        def __init__(self, module, loss_fn):
            super().__init__(module, loss_fn)

        def forward(self, input, target):
            output = self.module(input)
            return output, self.loss_fn(output, target)


    model = nn.Sequential(
        nn.Flatten(),
        PipeSplitWrapper(nn.Linear(28 * 28, 128)),
        PipeSplitWrapper(nn.ReLU()),
        PipeSplitWrapper(nn.Linear(128, 64)),
        PipeSplitWrapper(nn.ReLU()),
        PipeSplitWrapper(nn.Linear(64, 10))
    )

    wrapper = OutputLossWrapper(model, cross_entropy)
    wrapper.to(device)

    optimizer = optim.Adam(wrapper.parameters())

    loaders = {
        "train": train_dataloader,
        "valid": valid_dataloader
    }

    max_epochs = 10

    for epoch in range(max_epochs):
        print(f"Epoch: {epoch + 1}")
        epoch_correct = 0
        epoch_all = 0
        for k, dataloader in loaders.items():
            for i, (x_batch, y_batch) in enumerate(tqdm(dataloader) if USE_TQDM else dataloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
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
