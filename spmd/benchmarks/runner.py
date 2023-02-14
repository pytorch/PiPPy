# From https://fb.workplace.com/notes/1663126350733063
import argparse
from timeit import default_timer as timer
import os
import torch
import torch.multiprocessing as mp
import subprocess
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Replicate
from models import DemoConfig, MnistConfig
from spmd import Schema, SPMD
from spmd.compiler.graph_optimization import GraphOptimization

model_to_config = {"demo": DemoConfig(), "mnist": MnistConfig()}


def print0(pstr):
    if torch.distributed.get_rank() == 0:
        print(pstr)


def compute_nparams_and_flops(model, args):
    nparams = 0
    for name, params in model.named_parameters():
        nparams += params.nelement()
    nparams = torch.tensor(nparams).cuda()
    torch.distributed.all_reduce(nparams)
    nparams = nparams.item()
    tflops = 6.0 * nparams * args.batch_size * args.seq_size
    return nparams, tflops


def benchmark(args):
    model_config = model_to_config.get(args.model)
    train_data_loader, _ = model_config.get_train_and_test_data_loaders(args)

    if args.ddp:
        model = model_config.get_model()
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), find_unused_parameters=True)
        model._set_static_graph()
    elif args.spmd:
        device_type = "cuda"
        world_size = torch.distributed.get_world_size()
        device_mesh = DeviceMesh(device_type, list(range(world_size)))
        model = model_config.get_model()
        model = SPMD(
            model.cuda(),
            schema=Schema(
                mesh=device_mesh,
                placements=[Replicate()],
            ),
            optimizations=[GraphOptimization("fuse_communication_cat")],

        )

    else:
        print0("Running on a single GPU!")

    model.train()        
    nparams, tflops = compute_nparams_and_flops(model, args)
    print0(f'{nparams//1e6}M parameters, {tflops} FLOPS')

    def train_step(model, x, y, criterion):
        pred = model(x) 
        return criterion(pred, y)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()
    step, metric, t0 = 0, 0.0, timer()
    scale, log_freq = 1.0, args.log_freq
    total_tokens = 0
    total_time = 0.0
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=args.tb_wait, warmup=2, active=10, repeat=0),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/tmp/debug'),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    ) as torch_profiler:
        for batch_idx, data in enumerate(train_data_loader):
            x, y = data
            # TODO(anj): Make this part of the transform instead of this conditional.
            if args.model == "mnist":
                x = x.view(x.shape[0], -1)
            step += 1
            cuda_start.record()
            x, y = x.cuda(), y.cuda()
            loss = train_step(model.cuda(), x, y, criterion)
            loss.backward()
            cuda_end.record()
            if torch_profiler:
                torch_profiler.step()

            total_tokens += x.numel()
            torch.cuda.synchronize(torch.distributed.get_rank())
            total_time += cuda_start.elapsed_time(cuda_end) / 1000
            metric += loss.item() / log_freq
            optimizer.zero_grad()

            if step and step % log_freq == 0:
                print0(f'step: {step:6d}, wps_cuda: {total_tokens/(total_time):.3f} wps: {total_tokens/(timer() - t0):.3f} loss: {metric:.3f}, mem: {torch.cuda.max_memory_allocated()//1e9} GiB, t: {timer() - t0:.2f} sec')
                metric = 0.0
                total_tokens = 0
                total_time = 0.0
                t0 = timer()
            if step >= args.steps:
                break


def run_benchmark(rank, world_size, args):
    torch.manual_seed(0)
    setup(rank, world_size)
    if args.model in model_to_config.keys():
        benchmark(args)
    else:
        raise RuntimeError()

def setup(local_rank, world_size):
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    hostnames = subprocess.check_output(
         ["scontrol", "show", "hostnames", node_list]
     )
    master_host = hostnames.split()[0].decode("utf-8")

    nnodes = int(os.environ.get("SLURM_NNODES"))
    node_id = int(os.environ.get("SLURM_NODEID"))
    gpus_per_node = world_size

    if world_size == 1:
        rank = 0
    else:
        world_size = nnodes * gpus_per_node
        rank = node_id * gpus_per_node + local_rank

    backend = "nccl"
    PORT = 6001
    torch.distributed.init_process_group(
        backend,
        init_method=f"tcp://{master_host}:{PORT}",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(local_rank)


def run_dd(demo_fn, world_size, args):

    # os.environ["NCCL_DEBUG"] = "INFO"

    mp.spawn(
         demo_fn,
         args=(
             world_size,
             args,
         ),
         nprocs=world_size,
         join=True,
     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seq_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--spmd", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--tb_wait", type=int, default=1000)
    parser.add_argument("--compiler_backend", type=str)
    parser.add_argument("--model", type=str)

    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    n_gpus = min(args.gpus, n_gpus)
    assert n_gpus >= 2, f"SPMD requires at least 2 GPUs to run, found {n_gpus}"
    run_dd(run_benchmark, n_gpus, args)
