import argparse
# import logging
import os
import socket

import torch
import torch.distributed
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from pippy import Pipe, PipelineDriverFillDrain, annotate_split_points, PipeSplitWrapper
from pippy.microbatch import TensorChunkSpec, CustomReducer
from pippy.utils import tp_transports, get_argparser

# logging.getLogger().setLevel(logging.DEBUG)

USE_TQDM = True

# DIMS = [28 * 28, 100, 10]
# DP_LAYERS = 1
# PP_LAYERS = 1

# DIMS = [28 * 28, 300, 100, 30, 10]
# DP_LAYERS = 2
# PP_LAYERS = 2

DIMS = [28 * 28, 500, 250, 100, 50, 25, 10]
DP_LAYERS = 2
PP_LAYERS = 4

assert DP_LAYERS + PP_LAYERS == len(DIMS) - 1


class DDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.Sequential(
            *[layer for i in range(DP_LAYERS) for layer in [nn.Linear(DIMS[i], DIMS[i + 1], bias=True), nn.ReLU()]]
        )

    def forward(self, x):
        return self.mod(x)


class PipeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.Sequential(
            *[layer for i in range(DP_LAYERS, len(DIMS) - 1) for layer in
              [nn.Sequential(nn.Linear(DIMS[i], DIMS[i + 1], bias=True), nn.ReLU())]]
        )

    def forward(self, x, labels):
        logits = self.mod(x)
        loss = F.cross_entropy(logits, labels)
        return logits, loss


def resolve_pg_per_stage(pp_rank):
    assert dp_pg_per_pp_rank
    return dp_pg_per_pp_rank[pp_rank]


class DDP2PipeConnector(nn.Module):
    def __init__(self, pp_ranks, pp_group, dp_group_size, device):
        super().__init__()
        self.pp_ranks = pp_ranks
        self.pp_group = pp_group
        self.device = device

        self.last_grads = None

        rank = torch.distributed.get_rank()
        driver_rank = self.pp_ranks[0]

        if rank == driver_rank:
            self.pipe_model = PipeModel()

            for i in range(1, PP_LAYERS):
                annotate_split_points(self.pipe_model, {f'mod.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})

            # print(self.pipe_model)

            self.pipe = Pipe.from_tracing(self.pipe_model, output_loss_value_spec=(False, True))
            self.pipe.to(self.device)
            args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
            kwargs_chunk_spec = {}
            output_chunk_spec = (TensorChunkSpec(0), CustomReducer(torch.tensor(0.0), lambda a, b: a + b))
            all_worker_ranks = self.pp_ranks
            chunks = len(all_worker_ranks)

            # print(self.pipe)

            self.pipeline = PipelineDriverFillDrain(self.pipe, chunks,
                                                    args_chunk_spec, kwargs_chunk_spec, output_chunk_spec,
                                                    world_size=len(all_worker_ranks),
                                                    all_ranks=all_worker_ranks,
                                                    _debug_mask_minibatches=False)

            self.pipeline.init_data_parallel(dp_group_size, dp_pg_cb=resolve_pg_per_stage)

            self.optimizer = self.pipeline.instantiate_optimizer(optim.Adam)

    def forward(self, x, labels):
        rank = torch.distributed.get_rank()
        driver_rank = self.pp_ranks[0]

        if rank == driver_rank:
            x_list = [torch.empty_like(x) for _ in range(len(self.pp_ranks))]
            labels_list = [torch.empty_like(labels) for _ in range(len(self.pp_ranks))]
        else:
            x_list = None
            labels_list = None

        torch.distributed.gather(x, gather_list=x_list, dst=driver_rank, group=self.pp_group)
        torch.distributed.gather(labels, gather_list=labels_list, dst=driver_rank, group=self.pp_group)

        if rank == driver_rank:
            x_list = [x.detach() if x.grad_fn is not None else x for x in x_list]
            x_concat = torch.cat(x_list)
            x_concat.requires_grad = True
            labels_concat = torch.cat(labels_list)
            logits_concat, loss = self.pipeline(x_concat, labels_concat)
            logits_chunks = list(logits_concat.chunk(len(self.pp_ranks)))
        else:
            logits_chunks = None
            loss = torch.empty((), device=self.device)

        logits = torch.empty((x.size(0), DIMS[-1]), device=self.device)
        torch.distributed.scatter(logits, logits_chunks, src=driver_rank, group=self.pp_group)
        torch.distributed.broadcast(loss, src=driver_rank, group=self.pp_group)

        if torch.is_grad_enabled():
            if rank == driver_rank:
                last_grads_chunks = [x[0] for x in self.pipeline.last_grads]  # TODO
            else:
                last_grads_chunks = None
            last_grads = torch.empty_like(x)
            torch.distributed.scatter(last_grads, last_grads_chunks, src=driver_rank, group=self.pp_group)

            self.last_grads = last_grads

        return logits, loss

    def optimizer_zero_grad(self):
        rank = torch.distributed.get_rank()
        driver_rank = self.pp_ranks[0]
        if rank == driver_rank:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        rank = torch.distributed.get_rank()
        driver_rank = self.pp_ranks[0]
        if rank == driver_rank:
            self.optimizer.step()


class FullModel(nn.Module):
    def __init__(self, pp_ranks_per_dp_group, pp_pg_per_dp_group, device):
        super().__init__()
        self.flatten = nn.Flatten()
        self.ddp_model = DDPModel().to(device)
        self.ddp_model = DDP(self.ddp_model)

        rank = torch.distributed.get_rank()
        dp_group_size = len(pp_ranks_per_dp_group)
        self.pipe_model = DDP2PipeConnector(pp_ranks=pp_ranks_per_dp_group[rank % dp_group_size],
                                            pp_group=pp_pg_per_dp_group[rank % dp_group_size],
                                            dp_group_size=dp_group_size, device=device)

    def forward(self, x, labels):
        flatten = self.flatten(x)
        ddp_output = self.ddp_model(flatten)
        pipe_output = self.pipe_model(ddp_output, labels)
        return pipe_output, ddp_output


def run_master(args, pp_ranks_per_dp_group, pp_pg_per_dp_group):
    chunks = PP_LAYERS  # TODO
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

    model = FullModel(pp_ranks_per_dp_group=pp_ranks_per_dp_group,
                      pp_pg_per_dp_group=pp_pg_per_dp_group,
                      device=args.device)

    ddp_model_optimizer = optim.Adam(model.ddp_model.parameters())

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
                    model.train()
                    ddp_model_optimizer.zero_grad()
                    model.pipe_model.optimizer_zero_grad()
                    (outp, loss), ddp_outp = model(x_batch, y_batch)
                else:
                    model.eval()
                    with torch.no_grad():
                        (outp, _), _ = model(x_batch, y_batch)
                preds = outp.argmax(-1)
                correct = (preds == y_batch).sum()
                all = len(y_batch)
                epoch_correct += correct.item()
                epoch_all += all
                if k == "train":
                    # loss.backward()
                    ddp_outp.backward(gradient=model.pipe_model.last_grads)
                    model.pipe_model.optimizer_step()
                    ddp_model_optimizer.step()
            print(f"Loader: {k}. Accuracy: {epoch_correct / epoch_all}")


def run_worker(rank, args):
    args.rank = rank

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    actual_world_size = args.dp_group_size * args.pp_group_size

    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=512,
                                              rpc_timeout=1800,
                                              _transports=tp_transports())
    if args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
            for i in range(actual_world_size):
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
        else:
            args.cuda = 0
            print('Warning: no CUDA device found. Running on CPU instead.')

    args.device = f'cuda:{dev_id}' if args.cuda else 'cpu'
    print(f"rank = {rank} host/pid/device = "
          f"{socket.gethostname()}/{os.getpid()}/{args.device}")

    # Init DDP process group
    backend = "nccl" if args.cuda else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=actual_world_size)

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=actual_world_size,
        rpc_backend_options=options
    )

    global dp_pg_per_pp_rank
    dp_ranks_per_pp_rank = torch.arange(actual_world_size).reshape(args.pp_group_size,
                                                                   args.dp_group_size).tolist()
    dp_pg_per_pp_rank = [torch.distributed.new_group(ranks) for ranks in dp_ranks_per_pp_rank]

    pp_ranks_per_dp_group = [[i * args.dp_group_size + rank for i in range(args.pp_group_size)]
                             for rank in range(args.dp_group_size)]

    pp_pg_per_dp_group = [torch.distributed.new_group(ranks) for ranks in pp_ranks_per_dp_group]

    args.driver_group = torch.distributed.new_group(list(range(args.dp_group_size)))

    global exclude_master
    exclude_master = args.exclude_master if hasattr(args, 'exclude_master') else 0

    run_master(args, pp_ranks_per_dp_group, pp_pg_per_dp_group)
    rpc.shutdown()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 8)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))

    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=15)
    args = parser.parse_args()

    args.pp_group_size = PP_LAYERS

    assert args.world_size % args.pp_group_size == 0

    args.dp_group_size = args.world_size // args.pp_group_size

    if args.rank == -1:
        mp.spawn(run_worker, args=(args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args)
    else:
        print("I'm unused, exiting")
