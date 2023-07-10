from tqdm import tqdm
import argparse
import os

import torch
from torch import nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from pippy.IR import LossWrapper, PipeSplitWrapper
from pippy.microbatch import sum_reducer, TensorChunkSpec
from pippy.compile import compile_stage

USE_TQDM = bool(int(os.getenv("USE_TQDM", 1)))
LR_VERBOSE = bool(int(os.getenv("LR_VERBOSE", 1)))


def run_worker(args):
    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3017,), (0.3081))
    ])
    # load data
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    valid_data = datasets.MNIST("./data", train=False, transform=transform)
    # setup training sampler
    # train_sampler = DistributedSampler(train_data, num_replicas)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size * args.chunks)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size * args.chunks)

    # define custom loss wrapper
    class OutputLossWrapper(LossWrapper):
        def __init__(self, module, loss_fn):
            super().__init__(module, loss_fn)

        def forward(self, input, target):
            output = self.module(input)

            return output, self.loss_fn(output, target)

    # define model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        PipeSplitWrapper(
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
            )
        ),
        PipeSplitWrapper(nn.Linear(64, 10)),
    )

    # define model wrapper
    wrapper = OutputLossWrapper(model, cross_entropy)
    wrapper.to(args.device)

    # sample input
    x = torch.randint(0, 5, (args.batch_size * args.chunks, 28, 28), device=args.device)
    target = torch.randint(0, 9, (args.chunks * args.batch_size, ), device=args.device)

    output_chunk_spec = (TensorChunkSpec(0), sum_reducer)

    # setup compile stage
    stage = compile_stage(
        wrapper,
        rank=args.rank,
        num_ranks=args.world_size,
        num_chunks=args.chunks,
        device=args.device,
        group=None,
        example_inputs=[x, target],
        output_chunk_spec=output_chunk_spec,
    )

    # setup optimizer
    optimizer = optim.Adam(stage.submod.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    # setup lr scheduler
    lr_sched = optim.lr_scheduler.LinearLR(optimizer, verbose=LR_VERBOSE)

    loaders = {
        "train": train_dataloader,
        "valid": valid_dataloader,
    }

    batches_events_contexts = []

    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1} of {args.max_epochs}")

        for k, dataloader in loaders.items():
            epoch_correct = 0
            epoch_all = 0
            for i, (x_batch, y_batch) in enumerate(tqdm(dataloader) if USE_TQDM else dataloader):
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)

                if k == "train":
                    stage.train()
                    optimizer.zero_grad()

                    out = None

                    if args.rank == 0:
                        stage(x_batch)
                    elif args.rank == args.world_size - 1:
                        out = stage(y_batch)
                    else:
                        stage()

                    # outp, loss = stage(x_batch, y_batch)
                    if out:
                        preds = out.argmax(-1)
                        correct = (preds == y_batch).sum()
                        all = len(y_batch)
                        epoch_correct += correct.item()
                        epoch_all += all

                    optimizer.step()
                # else:
                #     # stage.eval()
                #     with torch.no_grad():
                #         if args.rank == 0:
                #             stage(x_batch, y_batch)
                #         elif args.rank == args.world_size - 1:
                #             out = stage()
                #         else:
                #             stage()
                #         # outp, _ = stage(x_batch, y_batch)

                #             preds = out.argmax(-1)
                #             correct = (preds == y_batch).sum()
                #             all = len(y_batch)
                #             epoch_correct += correct.item()
                #             epoch_all += all

            print(f"Loader: {k} Accuracy: {epoch_correct / epoch_all}")

            if k == "train":
                lr_sched.step()
                # if LR_VERBOSE:
                #     print(f"Pipe ")  # should we have pp_ranks


    dist.barrier()
    print(f"Rank {args.rank} completed!")


def main(args=None):
    # set seed for reproducibility
    torch.manual_seed(15)

    # set up parser
    parser = argparse.ArgumentParser()
    # set up arguments
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--visualize", type=int, default=1, choices=[0, 1])
    args = parser.parse_args(args)
    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # init process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"  # TODO: change to args.cuda after setting up args

    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run_worker(args)


if __name__ == "__main__":
    main()
