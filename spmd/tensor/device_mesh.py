# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.distributed.distributed_c10d import (
    get_rank,
    ReduceOp,
    _get_default_group,
    _reduce_scatter_base,
)

# autograd enabled collective
from torch.distributed.nn.functional import (
    all_gather,
    _all_gather_base,
    all_reduce,
    broadcast,
    scatter,
)

_global_device_mesh = None


def get_global_device_mesh():
    global _global_device_mesh
    return _global_device_mesh


def set_global_device_mesh(mesh):
    global _global_device_mesh
    _global_device_mesh = mesh


class DeviceMesh(object):
    """
    Device Mesh object, can be used as a context manager.
    By default describes the device ids, layout and serves
    as a proxy for communication among the device lists.

    We use the default ProcessGroup in this DeviceMesh class
    to implement proper communications. Note that we also
    add collective wrappers in this class. This is used to
    decouple detailed communication backend with the underlying
    DistributedTensor implementation.
    """

    device_type: str
    mesh: torch.Tensor
    # _world_pg: ProcessGroup

    def __init__(self, device_type, mesh):
        self.device_type = device_type
        self.mesh = torch.Tensor(mesh)

        default_pg = _get_default_group()
        backend_name = default_pg._get_backend_name()
        if device_type == "cpu":
            assert (
                backend_name == "gloo"
            ), f"ProcessGroup backend: {backend_name} not supporting CPU!"
        elif device_type == "cuda":
            assert backend_name == "gloo" or backend_name == "nccl"
        else:
            raise RuntimeError(
                f"DeviceMesh only support cpu or cuda device type, but got {device_type}"
            )

        # TODO: support multi-dimensional device mesh
        assert self.mesh.ndim == 1, "Only support 1-d device mesh for now"

    def __enter__(self):
        # set global device_mesh to this instance
        set_global_device_mesh(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # unset global device mesh
        set_global_device_mesh(None)

    def __repr__(self):
        return f"DeviceMesh:({self.mesh})"

    def size(self, dim=0):
        return self.mesh.size(dim)

    @property
    def ndim(self):
        return self.mesh.ndim

    def backend(self):
        return _get_default_group()._get_backend_name()

    def get_rank(self):
        return get_rank()

    def scatter(self, tensors, src=0) -> torch.Tensor:
        current_rank = get_rank()
        return scatter(tensors, src=src)

    def broadcast(self, tensor, src=0):
        return broadcast(tensor, src=src)

    def all_gather(self, tensor):
        return all_gather(tensor)

    def all_gather_base(self, output_tensor, tensor):
        # only nccl have all_gather base
        if self.backend() == "nccl":
            return _all_gather_base(output_tensor, tensor)
        else:
            # if not nccl, fallback to use all_gather
            # and reform the output tensor
            gathered_chunks = self.all_gather(tensor)
            # TODO: find more performant way
            for chunk in gathered_chunks:
                output_tensor.copy_(torch.cat(gathered_chunks))
            return output_tensor

    def all_reduce(self, tensor, op=ReduceOp.SUM):
        return all_reduce(tensor, op=op)

    def reduce_scatter_base(self, output, input, op=ReduceOp.SUM):
        # NOTE: two caveats:
        # 1. only NCCL support reduce_scatter
        # 2. we are using a non-autograd enabled reduce_scatter
        #    this is fine as we are using it now on redistribute
        #    which have backward enabled, but if we want to use
        #    this in other case which requires autograd, we should
        #    add the autograd enabled collective in distributed/nn/functional
        if self.backend() == "nccl":
            _reduce_scatter_base(output, input, op)
            return output
        else:
            # it's gloo, which does not have reduce_scatter
            # we have to do all_reduce + scatter
            reduced_tensor = self.all_reduce(input)
            shard_dim = 0
            num_chunks = 0
            for i in range(input.ndim):
                if input.size(i) > output.size(i):
                    num_chunks = input.size(i) // output.size(i)
                    shard_dim = i
                    break

            chunks = reduced_tensor.chunk(num_chunks, dim=shard_dim)
            return self.scatter(chunks)
