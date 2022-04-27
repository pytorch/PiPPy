# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import logging
import os
import socket

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms  # type: ignore
from tqdm import tqdm  # type: ignore

from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, LossWrapper
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.microbatch import CustomReducer, TensorChunkSpec

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

torch.fx.Tracer.proxy_buffer_attributes = True

USE_TQDM = os.getenv('USE_TQDM', True)

def run_master(args):
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)
    print("Using device:", args.device)

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

    pipe = Pipe.from_tracing(wrapper, MULTI_USE_PARAM_CONFIG, output_loss_value_spec=(False, True))
    pipe.to(args.device)

    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}
    output_chunk_spec = (TensorChunkSpec(0), CustomReducer(torch.tensor(0.0), lambda a, b: a + b))
    pipe_driver: PipelineDriverBase = schedules[args.schedule](pipe, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec,
                                                               len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                               _debug_mask_minibatches=True)

    optimizer = pipe_driver.instantiate_optimizer(optim.Adam, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

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
                x_batch = x_batch.to(args.device)
                y_batch = y_batch.to(args.device)
                if k == "train":
                    pipe_driver.train()
                    optimizer.zero_grad()
                    outp, _ = pipe_driver.run(chunks, x_batch, y_batch)
                    preds = outp.argmax(-1)
                    correct = (preds == y_batch).sum()
                    all = len(y_batch)
                    epoch_correct += correct.item()
                    epoch_all += all
                    optimizer.step()
                else:
                    pipe_driver.eval()
                    with torch.no_grad():
                        outp, _ = pipe_driver.run(chunks, x_batch, y_batch)
                        preds = outp.argmax(-1)
                        correct = (preds == y_batch).sum()
                        all = len(y_batch)
                        epoch_correct += correct.item()
                        epoch_all += all

            print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256,
                                              _transports=["shm", "uv"])
    if args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
            for i in range(world_size):
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
        else:
            args.cuda = 0
    args.device = f'cuda:{dev_id}' if args.cuda else 'cpu'
    print(f"rank = {rank} host/pid/device = "
          f"{socket.gethostname()}/{os.getpid()}/{args.device}")

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    if rank == 0:
        run_master(args)
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 7)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    args = parser.parse_args()
    args.world_size = 7  # "This program requires exactly 6 workers + 1 master"

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
