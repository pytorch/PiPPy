# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import pippy

from pippy.fx.passes import shape_prop
from pippy.IR import Pipe


def _make_tensor_from_meta(
    tensor_meta: shape_prop.TensorMetadata,
    device: torch.device,
):
    return torch.empty(
        tensor_meta.shape, dtype=tensor_meta.dtype, device=device
    )


class RecvInfo:
    def __init__(
        self,
        input_name: str,
        source: int,
        buffer: torch.Tensor,
    ):
        self.input_name = input_name
        self.source = source
        self.buffer = buffer


class PipelineStage(torch.nn.Module):
    def __init__(
        self,
        pipe: Pipe,
        rank: int,
        nstages: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
    ):
        super().__init__()
        self.pipe = pipe
        self.rank = rank
        self.nstages = nstages
        self.device = device
        self.group = group

        # Find my submodule
        self.split_gm = self.pipe.split_gm
        named_children = list(self.split_gm.named_children())
        self.name, self.submod = named_children[rank]
        logging.info(f"[{self.rank}][{self.name}] "
            f"Creating PipelineStage: {self.submod}"
        )

        # Find my node in graph
        self.node = None
        for node in self.split_gm.graph.nodes:
            if node.name == self.name:
                 self.node = node
                 break
        if not self.node:
            raise AssertionError(
                f"Cannot find {self.name} in graph"
            )

        self.create_recv_buffers()


    def create_recv_buffers(
        self,
    ):
        def create_recv_tensor(input_node):
            # TODO: do not create buffer for placeholder etc
            tensor_meta = input_node.meta["tensor_meta"]
            logging.info(f"[{self.rank}][{self.name}] "
                f"Creating recv buffer for input '{input_node.name}' : {tensor_meta.shape}"
            )
            return RecvInfo(
                input_node.name,
                self.rank - 1,  #TODO: find source rank
                _make_tensor_from_meta(tensor_meta, self.device),
            )

        # Tuple[Tuple[source, tensor]]
        self.args_recv_info = pippy.fx.node.map_arg(
            self.node.args, create_recv_tensor
        )

        # Dict[keyword, Tuple[source, tensor]]
        self.kwargs_recv_info = pippy.fx.node.map_arg(
            self.node.kwargs, create_recv_tensor
        )


    def forward(self, *args, **kwargs):
        def recv_tensor(info):
            if isinstance(info, RecvInfo):
                logging.info(f"[{self.rank}][{self.name}] "
                    f"Receiving tensor '{info.input_name}' from Rank {info.source}"
                )
                dist.recv(info.buffer, info.source, group=self.group)
                return info.buffer
            else:
                return info

        if not args:
            args = pippy.fx.node.map_aggregate(
                self.args_recv_info, recv_tensor,
            )

        if not kwargs:
            kwargs = pippy.fx.node.map_aggregate(
                self.kwargs_recv_info, recv_tensor,
            )

        output = self.submod(*args, **kwargs)

        dist.send(
            output,
            (self.rank + 1) % self.nstages,  #TODO
            group=self.group
        )

        return output
            