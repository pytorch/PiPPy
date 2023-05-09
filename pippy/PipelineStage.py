# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from pippy.fx.passes import shape_prop
from pippy.IR import Pipe


def _make_tensor_from_meta(
    tensor_meta: shape_prop.TensorMetadata,
    device: torch.device,
):
    return torch.empty(
        tensor_meta.shape, dtype=tensor_meta.dtype, device=device
    )


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
        logging.info(f"[{self.rank}] "
            f"Creating PipelineStage {self.name}: {self.submod}"
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

        # name : (source, tensor)
        self.recv_buffers: Dict[str, Tuple[int, torch.tensor]] = {}
        self.create_recv_buffers()


    def create_recv_buffers(
        self,
    ):
        input = self.node.args[0]  # args is a tuple
        tensor_meta = input.meta["tensor_meta"]
        logging.info(f"[{self.rank}] "
            f"Creating recv buffer for {input.name} : {tensor_meta}"
        )
        self.recv_buffers.setdefault(
            input.name,
            (
                self.rank - 1,  #TODO
                _make_tensor_from_meta(tensor_meta, self.device),
            )
        )


    def forward(self, *args, **kwargs):
        if not args:
            inputs = []
            for name, recv_info in self.recv_buffers.items():
                (src, buf) = recv_info
                logging.info(f"[{self.rank}] "
                    f"Receiving {name} from Rank {src}"
                )
                dist.recv(buf, src, group=self.group)
                inputs.append(buf)
        else:
            inputs = args

        output = self.submod(*inputs)

        dist.send(
            output,
            (self.rank + 1) % self.nstages,  #TODO
            group=self.group
        )

        return output
            