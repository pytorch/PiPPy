# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

import pippy
import pippy.fx

import torch
import torch.distributed as dist

from pippy.SaveModule import save_checkpoint
from pippy.IR import LossWrapper, PipeSplitWrapper
from pippy.microbatch import sum_reducer, TensorChunkSpec

from torch import nn, optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms  # type: ignore
from tqdm import tqdm  # type: ignore


pippy.fx.Tracer.proxy_buffer_attributes = True

USE_TQDM = bool(int(os.getenv("USE_TQDM", "1")))


# Get process group for ranks in a pipeline
def get_pp_subgroup(args):
    my_pp_rank = args.rank // args.dp_group_size
    my_dp_rank = args.rank % args.dp_group_size
    for dp_rank in range(0, args.dp_group_size):
        pp_group_ranks = list(
            range(dp_rank, args.world_size, args.dp_group_size)
        )
        pp_group = dist.new_group(ranks=pp_group_ranks)
        if dp_rank == my_dp_rank:
            my_pp_group = pp_group
    print(f"Rank {args.rank} done getting pp group")
    return my_pp_group, my_pp_rank


# Get DP process group for ranks with the same stage
def get_dp_subgroup(args):
    my_pp_rank = args.rank // args.dp_group_size
    my_dp_rank = args.rank % args.dp_group_size
    for pp_rank in range(0, args.pp_group_size):
        dp_group_ranks = list(
            range(
                pp_rank * args.dp_group_size, (pp_rank + 1) * args.dp_group_size
            )
        )
        dp_group = dist.new_group(ranks=dp_group_ranks)
        if pp_rank == my_pp_rank:
            my_dp_group = dp_group
    print(f"Rank {args.rank} done getting dp group")
    return my_dp_group, my_dp_rank


def run_worker(args):
    torch.manual_seed(42)

    # Get DP and PP sub process groups
    dp_group, dp_rank = get_dp_subgroup(args)
    pp_group, pp_rank = get_pp_subgroup(args)

    batch_size = args.batch_size * args.chunks

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_data = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    valid_data = datasets.MNIST("./data", train=False, transform=transform)

    train_sampler = DistributedSampler(
        train_data,
        num_replicas=args.dp_group_size,
        rank=dp_rank,
        shuffle=False,
        drop_last=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size
    )

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
        PipeSplitWrapper(
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
            )
        ),
        PipeSplitWrapper(nn.Linear(64, 10)),
    )

    wrapper = OutputLossWrapper(model, cross_entropy)

    wrapper.to(args.device)

    output_chunk_spec = (TensorChunkSpec(0), sum_reducer)

    # sample input
    x = torch.randint(0, 5, (batch_size, 28, 28), device=args.device)
    target = torch.randint(0, 9, (batch_size,), device=args.device)

    stage = pippy.compile_stage(
        wrapper,
        pp_rank,
        args.pp_group_size,
        args.chunks,
        args.device,
        pp_group,
        [x, target],
        output_chunk_spec=output_chunk_spec,
    )

    optimizer = optim.Adam(
        stage.submod.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8
    )
    # TODO: add back LR scheduler
    # lr_sched = pipe_driver.instantiate_lr_scheduler(optim.lr_scheduler.LinearLR, verbose=LR_VERBOSE)

    loaders = {"train": train_dataloader, "valid": valid_dataloader}

    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")
        for k, dataloader in loaders.items():
            epoch_correct = 0
            epoch_all = 0
            for i, (x_batch, y_batch) in enumerate(
                tqdm(dataloader) if USE_TQDM else dataloader
            ):
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)
                if k == "train":
                    outp = None
                    optimizer.zero_grad()
                    if pp_rank == 0:
                        stage(x_batch)
                    elif pp_rank == args.pp_group_size - 1:
                        outp, _ = stage(y_batch)
                    else:
                        stage()
                    optimizer.step()

                    if outp is not None:
                        preds = outp.argmax(-1)
                        correct = (preds == y_batch).sum()
                        all = len(y_batch)
                        epoch_correct += correct.item()
                        epoch_all += all

                    # save checkpoint - after training epoch
                    if (epoch + 1) % args.checkpoint_epochs == 0:
                        save_checkpoint(
                            stage,
                            checkpoint_dir=os.path.join(
                                "checkpoints", str(epoch + 1)
                            ),
                            optimizer=optimizer,
                        )
                else:
                    # TODO: add evaluation support in PiPPy
                    pass

            if pp_rank == args.pp_group_size - 1 and epoch_all > 0:
                print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")

            # if k == "train":
            #    lr_sched.step()

    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 3))
    )
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)

    parser.add_argument(
        "--replicate", type=int, default=int(os.getenv("REPLICATE", "0"))
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument("--visualize", type=int, default=0, choices=[0, 1])
    parser.add_argument("--checkpoint", type=int, default=0, choices=[0, 1])
    parser.add_argument("--checkpoint_epochs", type=int, default=5)
    parser.add_argument(
        "--chunks",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    args.pp_group_size = 3
    assert args.world_size % args.pp_group_size == 0
    args.dp_group_size = args.world_size // args.pp_group_size

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
