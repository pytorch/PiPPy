import argparse
import os

import torch
from torch import nn
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.nn.functional import cross_entropy
from torch.utils.data import DistributedSampler, DataLoader

from pippy.microbatch import sum_reducer, TensorChunkSpec
from pippy.IR import LossWrapper, PipeSplitWrapper
from pippy.compile import compile_stage


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

    # setup compile stage
    stage = compile_stage(
        wrapper,
        rank=args.rank,
        num_ranks=args.world_size,
        num_chunks=args.chunks,
        device=args.device,
        group=None,
        example_inputs=[x, target],
        # output_chunk_spec={
        #     "loss": sum_reducer,
        #     "logits": TensorChunkSpec(0),
        # },
    )

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
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--chunks", type=int, default=4)
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
