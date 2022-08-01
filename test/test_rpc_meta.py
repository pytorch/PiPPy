import argparse
import os
import socket

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef
import torch._C._distributed_rpc

from test_commons import tp_transports  # type: ignore


def _meta_tensor_method(tensors, device_types, sizes, numels, dtypes, strides, return_rrefs):
    tensors = [t.to_here() if isinstance(t, torch._C._distributed_rpc.PyRRef) else t for t in tensors]
    for tensor, device_type, size, numel, dtype, stride in zip(tensors, device_types, sizes, numels, dtypes,
                                                               strides):
        assert tensor.device.type == device_type, f"{tensor.device.type} vs {device_type}"
        assert tensor.size() == size, f"{tensor.size} vs {size}"
        assert tensor.numel() == numel, f"{tensor.numel()} vs {numel}"
        assert tensor.dtype == dtype, f"{tensor.dtype} vs {dtype}"
        assert tensor.stride() == stride, f"{tensor.stride()} vs {stride}"
    return [RRef(t) if return_rrefs else t for t in tensors]


def _test_meta_tensor(to, tensors, device_types, sizes, numels, dtypes, strides, return_rrefs):
    returned_tensors = rpc.rpc_sync(to, _meta_tensor_method,
                                    args=(tensors, device_types, sizes, numels, dtypes, strides, return_rrefs))
    returned_tensors = [t.to_here() if isinstance(t, torch._C._distributed_rpc.PyRRef) else t for t in
                        returned_tensors]
    for tensor, device_type, size, numel, dtype, stride in zip(returned_tensors, device_types, sizes, numels,
                                                               dtypes, strides):
        assert isinstance(tensor, torch.Tensor), f"{type(tensor)}"
        assert tensor.device.type == device_type
        assert tensor.size() == size
        assert tensor.numel() == numel
        assert tensor.dtype == dtype
        assert tensor.stride() == stride


def run_master(args):
    for device in ['cpu', 'cuda'] if args.cuda == 1 else ['cpu']:
        for return_rrefs in [False, True]:
            _test_meta_tensor("worker1",
                              [torch.empty(42, device='meta'),
                               torch.empty(6, 7, device=device, dtype=torch.bool),
                               RRef(torch.empty(2, 21, device='meta', dtype=torch.int)),
                               RRef(torch.empty(3, 14, device=device, dtype=torch.long))],
                              ['meta', device, 'meta', device],
                              [torch.Size((42,)), torch.Size((6, 7)), torch.Size((2, 21)), torch.Size((3, 14))],
                              [42, 42, 42, 42],
                              [torch.float, torch.bool, torch.int, torch.long],
                              [(1,), (7, 1), (21, 1), (14, 1)],
                              return_rrefs=return_rrefs)


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256,
                                              rpc_timeout=1800,
                                              _transports=tp_transports())
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
    # parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 2)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    args = parser.parse_args()
    args.world_size = 2

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
