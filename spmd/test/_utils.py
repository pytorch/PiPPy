# Copyright (c) Meta Platforms, Inc. and affiliates
import itertools
import sys
from functools import wraps
from typing import Any, Callable, Tuple, Dict, Optional, List

import torch
import torch.distributed as dist

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
)

from spmd import DeviceMesh, Placement, distribute_tensor, Shard, Replicate
from spmd.tensor.api import DTensor

# default GPU test size/world size
TEST_GPU_NUM = 4


class DistTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return TEST_GPU_NUM

    def init_pg(self, backend: str = "nccl") -> None:
        if backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if backend not in ["nccl", "gloo", "mpi"]:
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
        )

        # set device for nccl pg for collectives
        if backend == "nccl":
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()


# wrapper to initialize comms (processgroup)
def with_comms(
    func: Optional[  # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        Callable
    ] = None,
    backend: Optional[str] = None,
) -> Optional[  # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    Callable
]:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore
    ) -> None:
        # if backend not specified, and cuda available, then use nccl, else gloo
        pg_backend = (
            "nccl" if backend is None and torch.cuda.is_available() else "gloo"
        )
        if pg_backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.device_type = "cuda" if pg_backend == "nccl" else "cpu"
        self.init_pg(backend=pg_backend)
        func(self)  # type: ignore
        self.destroy_pg()

    return wrapper


# This is a class for converting args/kwargs of an op into distributed args/kwargs
class DTensorConverter(object):
    def __init__(self, mesh, args, kwargs):
        self.hit = 0
        self.miss = 0
        self.mesh = mesh
        self.args = args
        self.kwargs = kwargs
        self.flatten_args, self.flatten_args_spec = tree_flatten(args)
        self.flatten_kwargs, self.flatten_kwargs_spec = tree_flatten(kwargs)

        choices_for_args = []
        for arg in self.flatten_args:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        for arg in self.flatten_kwargs:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        self.sharding_combs = iter(itertools.product(*choices_for_args))
        # if dist.get_rank() == 0:
        #     print(f">>> len args: {len(args)}, len kwargs: {len(kwargs)}")
        #     for comb in self.sharding_combs:
        #         print(f">>>>> sharding comb: {comb}")

    def successful(self):
        return self.hit > 0 and self.miss == 0

    def is_supported_tensor(self, t: torch.Tensor):
        # TODO: dist tensor need to support quantized and sparse
        # tensors, quantized tensor might be relatively easy, but
        # sparse tensor have special layouts that we need to possibly
        # deal with, until we are clear about them, we don't officially
        # support them.
        return not any(
            [
                t.is_sparse_csr,
                t.is_sparse,
                t.is_mkldnn,
                t.is_quantized,
                t.is_nested,
                torch._is_functional_tensor(t),
                t.is_neg(),
                t.is_conj(),
                t.device.type in ("lazy", "meta"),
                # We need a way to test if a tensor is batched but there
                # is no official APi to do it
                # torch._C._is_batched(t),
            ]
        )

    def gen_sharding_choices_for_arg(self, arg: torch.Tensor):
        # NOTE we assume cube mesh here to simplify things
        mesh_size = self.mesh.size()
        if arg.dtype == torch.bool:
            # c10d collective does not support bool tensor
            # for bool tensor we treat it as replicated
            sharding_choices = [Replicate()] * mesh_size
            # set a field in tensor to tell the converter to
            # not run mesh.broadcast, we assume op with bool tensor
            # are the same tensor so we don't need to broadcast
            # TODO: add bool tensor support in c10d collective
            arg._no_comm = True
        else:
            # only generating choices with: replicate, or sharding
            # evenly on a dimension that could be sharded
            sharding_choices = [Replicate()] + [
                Shard(i)
                for i, s, in enumerate(arg.shape)
                if s > 1 and s % mesh_size == 0
            ]
        # TODO: add multi mesh choices
        # all_choices = itertools.product(
        #     *(self.mesh.ndim * [sharding_choices])
        # )
        return sharding_choices

    def __iter__(self):
        return self

    def __next__(self):
        try:
            next_sharding_choices = next(self.sharding_combs)
            idx = 0

            new_args = []
            for arg in self.flatten_args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(
                        self.to_dist_tensor(
                            arg, self.mesh, [next_sharding_choices[idx]]
                        )
                    )
                    idx += 1
                else:
                    new_args.append(arg)

            new_kwargs = []
            for arg in self.flatten_kwargs:
                if isinstance(arg, torch.Tensor):
                    new_kwargs.append(
                        self.to_dist_tensor(
                            arg, self.mesh, [next_sharding_choices[idx]]
                        )
                    )
                    idx += 1
                else:
                    new_kwargs.append(arg)

            return tree_unflatten(
                new_args, self.flatten_args_spec
            ), tree_unflatten(new_kwargs, self.flatten_kwargs_spec)
        except StopIteration:
            raise StopIteration

    def to_dist_tensor(
        self,
        t: torch.Tensor,
        mesh: DeviceMesh,
        placements: List[Placement],
        no_comm=False,
    ):
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            if self.is_supported_tensor(t):
                self.hit += 1
                if hasattr(t, "_no_comm"):
                    r = DTensor(
                        t, mesh, placements, requires_grad=t.requires_grad
                    )
                else:
                    r = distribute_tensor(t, mesh, placements)
                if type(t) is torch.nn.Parameter:
                    r = torch.nn.Parameter(r, requires_grad=r.requires_grad)
                return r
            else:
                self.miss += 1
                return t
        elif torch.overrides.is_tensor_like(t):
            # Blindly converting tensor subclasses to dist tensor can cause
            # unpredictable problems, we explicitly disable this conversion
            # for now.
            self.miss += 1
            return t
        else:
            raise RuntimeError(
                f"Trying to convert to DTensor, but got {type(t)}"
            )
