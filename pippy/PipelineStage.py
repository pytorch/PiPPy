# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import pippy

from pippy.fx.passes import shape_prop
from pippy.IR import Pipe
from pippy.microbatch import merge_chunks, split_args_kwargs_into_chunks


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
        chunks: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
        return_to_0: bool = False,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
    ):
        super().__init__()
        self.pipe = pipe
        self.rank = rank
        self.nstages = nstages
        self.chunks = chunks
        self.device = device
        self.group = group
        self.return_to_0 = return_to_0
        self.args_chunk_spec = args_chunk_spec
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.output_chunk_spec = output_chunk_spec

        # Find my submodule
        self.split_gm = self.pipe.split_gm
        named_children = list(self.split_gm.named_children())
        self.name, self.submod = named_children[rank]
        logging.info(
            f"[{self.rank}][{self.name}] "
            f"Creating PipelineStage:\n"
            f"{self.submod}"
        )

        # Find my node in graph
        self.node = None
        for node in self.split_gm.graph.nodes:
            if node.name == self.name:
                self.node = node
                break
        if not self.node:
            raise AssertionError(f"Cannot find {self.name} in graph")

        # Create submod name to rank mapping
        self.submod_to_rank: Dict[str, int] = {}
        for i, (name, _) in enumerate(self.split_gm.named_children()):
            self.submod_to_rank.setdefault(name, i)

        # Prepare send/recv infrastructure
        self._create_recv_buffers()
        self._create_send_info()

    def get_rank_of_submod(
        self,
        submod_name: str,
    ):
        try:
            return self.submod_to_rank[submod_name]
        except:
            raise AssertionError(f"Rank of {submod_name} not found")

    def _create_recv_buffers(
        self,
    ):
        def create_recv_tensor(
            input_node,
            output_idx: Optional[int] = None,
        ):
            """
            Create a tensor for receiving the `output_idx`-th value from
            `input_node`
            """
            if input_node.op == "placeholder":
                # Do not create buffer for placeholder
                return None

            # In case the input is a `getitem` node, we recursively find the
            # real source e.g. getitem1 = submod0[1]
            # Here `submod0` is args[0], 1 is args[1]
            if input_node.target is operator.getitem:
                if "tensor_meta" in input_node.meta:
                    real_input_node = input_node.args[0]
                    out_idx = input_node.args[1]
                    return create_recv_tensor(real_input_node, out_idx)
                else:
                    return None

            if output_idx is not None:
                # If a node has multiple output values, "tensor_meta" is a list
                # of tensor meta
                tensor_meta = input_node.meta["tensor_meta"][output_idx]
            else:
                tensor_meta = input_node.meta["tensor_meta"]

            logging.info(
                f"[{self.rank}][{self.name}] "
                f"Creating recv buffer for input '{input_node.name}' "
                f"value index {output_idx}: {tensor_meta.shape}"
            )

            src_rank = self.get_rank_of_submod(input_node.name)
            return RecvInfo(
                input_node.name,
                src_rank,
                _make_tensor_from_meta(tensor_meta, self.device),
            )

        # `args` is a Tuple, hence we will have:
        # Tuple[RecvInfo]
        self.args_recv_info = pippy.fx.node.map_arg(
            self.node.args, create_recv_tensor
        )

        # `kwargs` is a Dict, hence we will have:
        # Dict[keyword, RecvInfo]
        self.kwargs_recv_info = pippy.fx.node.map_arg(
            self.node.kwargs, create_recv_tensor
        )

    def _create_send_info(self):
        # Find send destinations
        def find_dst_rank(user) -> Optional[int]:
            if user.op == "output":
                if self.return_to_0:
                    # Send result back to pp rank 0
                    return 0
                else:
                    return None
            else:
                # User is a stage (`call_module`)
                return self.get_rank_of_submod(user.name)

        # Output index: List of receivers
        self.dst_infos: Dict[int, List] = {}
        out_idx = 0

        for user in self.node.users:
            if user.target is operator.getitem:
                # Recursively find the real destination
                gi_dsts = self.dst_infos.setdefault(out_idx, [])
                for gi_user in user.users:
                    gi_dsts.append(find_dst_rank(gi_user))
                # Next `getitem` will point to the next output index
                out_idx += 1
            else:
                # In case of single output value, `out_idx` will not increase
                dsts = self.dst_infos.setdefault(out_idx, [])
                dsts.append(find_dst_rank(user))

        logging.info(
            f"[{self.rank}][{self.name}] " f"Send info: {self.dst_infos}"
        )

    def forward(self, *args, **kwargs):
        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self.chunks,
                self.args_chunk_spec,
                self.kwargs_chunk_spec,
            )

        # Receive requests of a chunk
        recv_reqs : List[dist.Work] = []
        # Send requests of a chunk
        send_reqs : List[dist.Work] = []

        def recv_tensor(info):
            if isinstance(info, RecvInfo):
                logging.info(
                    f"[{self.rank}][{self.name}] "
                    f"Receiving tensor '{info.input_name}' from Rank {info.source}"
                )
                # Use async to parallelize recv of tensors
                work = dist.irecv(
                    info.buffer,
                    info.source
                    if self.group is None
                    else dist.get_global_rank(self.group, info.source),
                    group=self.group,
                )
                recv_reqs.append(work)
                return info.buffer
            else:
                return info

        output_chunks = []

        for chunk in range(self.chunks):
            recv_reqs.clear()
            if args:
                chunk_args = args_split[chunk]
            else:
                chunk_args = pippy.fx.node.map_aggregate(
                    self.args_recv_info,
                    recv_tensor,
                )

            if kwargs:
                chunk_kwargs = kwargs_split[chunk]
            else:
                chunk_kwargs = pippy.fx.node.map_aggregate(
                    self.kwargs_recv_info,
                    recv_tensor,
                )

            # Wait for all recvs to finish
            for work in recv_reqs:
                work.wait()

            # Compute
            output = self.submod(*chunk_args, **chunk_kwargs)

            # Unify output form to tuple for easy correspondance with
            # `dst_infos`
            output_tuple = output if isinstance(output, Tuple) else (output,)

            for idx, out in enumerate(output_tuple):
                dst_ranks = self.dst_infos[idx]
                for dst in dst_ranks:
                    if dst is None:
                        # If dst is a `output` node, we don't need to send
                        # (unless `return_to_0` is required)
                        continue
                    work = dist.isend(
                        out,
                        dst
                        if self.group is None
                        else dist.get_global_rank(self.group, dst),  # TODO
                        group=self.group,
                    )
                    send_reqs.append(work)

            output_chunks.append(output)

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in send_reqs:
            work.wait()

        # Last rank return merged results per original format
        if self.rank == self.nstages - 1:
            return merge_chunks(
                output_chunks,
                self.output_chunk_spec,
            )
        else:
            return None
