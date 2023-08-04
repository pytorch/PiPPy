# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import pippy
import pippy.fx
from pippy.backward import stage_backward, sync_barrier
from pippy.debug import map_debug_info

from pippy.fx.passes import shape_prop
from pippy.IR import Pipe
from pippy.microbatch import merge_chunks, split_args_kwargs_into_chunks
from pippy.utils import flatten_args


def _make_tensor_from_meta(
    tensor_meta: shape_prop.TensorMetadata,
    device: torch.device,
) -> torch.Tensor:
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

    def __repr__(self):
        return f"RecvInfo(input={self.input_name}, source={self.source}, buffer={self.buffer.size()})"


class StageArgPlaceholder:
    pass


class PipelineStage(torch.nn.Module):
    def __init__(
        self,
        pipe: Pipe,
        rank: int,
        nstages: int,
        chunks: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
        nstreams=2,
        global_depth=None,
    ):
        super().__init__()
        self.pipe = pipe
        self.rank = rank
        self.nstages = nstages
        self.chunks = chunks
        self.device = device
        self.group = group
        self.args_chunk_spec = args_chunk_spec
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.output_chunk_spec = output_chunk_spec
        self.nstreams = nstreams

        self.streams = []
        for i in range(nstreams):
            self.streams.append(torch.cuda.Stream())

        # inner pipelining
        if global_depth is not None:
            self.global_depth = global_depth
            self.inner_depth = global_depth // nstages
        else:
            self.global_depth = nstages
            self.inner_depth = 1

        # Find my submodule
        self.split_gm = self.pipe.split_gm
        named_children = list(self.split_gm.named_children())

        # submod = first inner node of each rank
        self.name, self.submod = named_children[rank * self.inner_depth]
        self.names, self.submods = [], []
        for i in range(self.inner_depth):
            name, submod = named_children[rank * self.inner_depth + i]
            self.names.append(name)
            self.submods.append(submod)

        logging.info(
            f"[{self.rank}][{self.name}] "
            f"Creating PipelineStage:\n"
            f"{self.submod}"
        )

        # Find my forward node in graph
        found_node = False
        for node in self.split_gm.graph.nodes:
            if node.name == self.name:
                self.node = node
                found_node = True
                break
        if not found_node:
            raise AssertionError(f"Cannot find {self.name} in graph")

        if (
            self.inner_depth > 1
        ):  # when inner pipelining is enabled, we have multiple nodes for this rank
            self.nodes = []
            for node in self.split_gm.graph.nodes:
                if node.name in self.names:
                    self.nodes.append(node)
            if len(self.nodes) == 0:
                raise AssertionError(f"Cannot find {self.names} in graph")

        # Find my backward node in graph
        if self.pipe.has_loss_and_backwards:
            found_bwd = False
            seen_bwd = -1
            for node in reversed(self.split_gm.graph.nodes):
                if (node.op, node.target) == ("call_function", stage_backward):
                    seen_bwd += 1
                    if seen_bwd == self.rank:
                        found_bwd = True
                        self.bwd_node = node
                        break
            if not found_bwd:
                raise AssertionError(
                    f"Cannot find backward for {self.name} in graph"
                )

        if (
            self.inner_depth > 1
        ):  # when inner pipelining is enabled, we have multiple bwd nodes for this rank
            self.bwd_nodes = []
            seen_bwd = -1
            added_bwd = 0
            for node in reversed(self.split_gm.graph.nodes):
                if (node.op, node.target) == ("call_function", stage_backward):
                    seen_bwd += 1
                    if seen_bwd // self.inner_depth == self.rank:
                        self.bwd_nodes.append(node)
                        added_bwd += 1
                        if added_bwd == self.inner_depth:
                            break
            if len(self.bwd_nodes) == 0:
                raise AssertionError(
                    f"Cannot find backward for {self.names} in graph"
                )

        # Create submod to rank mapping
        self.submod_to_rank: Dict[str, int] = {}
        for i, (name, _) in enumerate(self.split_gm.named_children()):
            self.submod_to_rank.setdefault(name, i // self.inner_depth)

        # Prepare send/recv infrastructure
        self._prepare_send_recv_infra()

    def _prepare_send_recv_infra(self):
        """
        Create send/recv infrastructures for activations (during forward) and
        gradients (during backward)
        """
        # chunk : Tuple of arg buffers
        self.args_recv_info: Dict[int, Tuple] = {}
        # chunk : Dict of kwarg buffers
        self.kwargs_recv_info: Dict[int, Dict] = {}
        for chunk in range(self.chunks):
            (
                self.args_recv_info[chunk],
                self.kwargs_recv_info[chunk],
            ) = self._create_act_recv_buffers()

        # Send info during forward for each activation
        self.act_send_info = self._create_act_send_info()

        if self.pipe.has_loss_and_backwards:
            # chunk : List of output grad buffers
            # `grad_recv_info` is a mirror of `act_send_info`
            self.grad_recv_info: Dict = {}
            for chunk in range(self.chunks):
                self.grad_recv_info[chunk] = self._create_grad_recv_info(
                    self.act_send_info
                )

            # Send info for input grads during backward
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info` + `kwargs_recv_info`
            self.grad_send_info = self._create_grad_send_info(
                self.args_recv_info[0],
                self.kwargs_recv_info[0],
            )

    def get_rank_of_submod(
        self,
        submod_name: str,
    ):
        if submod_name not in self.submod_to_rank:
            raise AssertionError(f"Rank of {submod_name} not found")

        return self.submod_to_rank[submod_name]

    def _create_act_recv_buffers(
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
                return StageArgPlaceholder()

            # In case the input is a `getitem` node, we recursively find the
            # real source e.g. getitem1 = submod0[1]
            # Here `submod0` is args[0], 1 is args[1]
            if input_node.target is operator.getitem:
                if "tensor_meta" in input_node.meta:
                    real_input_node = input_node.args[0]
                    out_idx = input_node.args[1]
                    return create_recv_tensor(real_input_node, out_idx)
                else:
                    raise NotImplementedError(
                        f"getitem gets a non-Tensor value, this is not yet supported. "
                        f"Node: {input_node.format_node()}"
                    )

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
            buffer = _make_tensor_from_meta(tensor_meta, self.device)
            # Enable gradient in training mode
            if self.pipe.has_loss_and_backwards:
                buffer.requires_grad_(True)
            return RecvInfo(
                input_node.name,
                src_rank,
                buffer,
            )

        # `args` is a Tuple, hence we will have:
        # Tuple[RecvInfo]
        args_recv_info = pippy.fx.node.map_arg(
            self.node.args, create_recv_tensor
        )

        # `kwargs` is a Dict, hence we will have:
        # Dict[keyword, RecvInfo]
        kwargs_recv_info = pippy.fx.node.map_arg(
            self.node.kwargs, create_recv_tensor
        )

        logging.info(
            f"[{self.rank}][{self.name}] "
            f"Activation recv info: {args_recv_info}"
        )
        return args_recv_info, kwargs_recv_info

    def find_dst_rank(
        self,
        user: pippy.fx.Node,
    ) -> Optional[int]:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        if user.op == "call_module":
            # User is a stage (`call_module`)
            return self.get_rank_of_submod(user.name)
        elif user.target is sync_barrier:
            # Send result back to pp rank 0
            return 0
        else:
            # - If user.op == "output":
            #   No need to send back to rank 0
            # - If user.target is stage_backward:
            #   No need to send assuming submod output is stored locally or
            #   should be re-calucated in case of activation checkpointing
            return None

    def _create_act_send_info(self):
        # Output index: List of receiver ranks
        act_send_info: Dict[int, List] = {}
        out_idx = 0

        if self.inner_depth == 1:
            node = self.node
        else:
            node = self.nodes[-1] # last node result is sent

        for user in node.users:
            if user.target is operator.getitem:
                # Recursively find the real destination
                gi_dsts = act_send_info.setdefault(out_idx, [])
                print(f'[Rank{self.rank}] gi_dsts{gi_dsts}')
                for gi_user in user.users:
                    dst_rank = self.find_dst_rank(gi_user)
                    print(
                        f"[Rank{self.rank}] what is dst rank of user {gi_user}? {dst_rank}"
                    )
                    if dst_rank is not None:
                        gi_dsts.append(dst_rank)
                # Next `getitem` will point to the next output index
                out_idx += 1
            else:
                # In case of single output value, `out_idx` will not increase
                print(f'[Rank{self.rank}] or do we meet here?')
                dsts = act_send_info.setdefault(out_idx, [])
                dst_rank = self.find_dst_rank(user)
                if dst_rank is not None:
                    dsts.append(dst_rank)
                print(f"[Rank{self.rank}] dest- {dsts}")

        logging.info(
            f"[{self.rank}][{self.name}] " f"Send info: {act_send_info}"
        )
        return act_send_info

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Dict[int, RecvInfo]:
        # Dict[output_index, RecvInfo]
        grad_recv_info: Dict = {}
        my_tensor_meta = self.node.meta["tensor_meta"]

        for out_idx, dst_list in act_send_info.items():
            if not dst_list:
                # No actual receiver for activation so no grad coming back
                continue

            # TODO: clean way
            if len(act_send_info) > 1:
                tensor_meta = my_tensor_meta[out_idx]
            else:
                tensor_meta = my_tensor_meta

            # TODO: otherwise needs grad accumulation
            assert len(dst_list) == 1
            grad_src = dst_list[0]
            grad_recv_info[out_idx] = RecvInfo(
                f"{grad_src}",
                grad_src,
                _make_tensor_from_meta(tensor_meta, self.device),
            )

        logging.info(
            f"[{self.rank}][{self.name}] " f"Grad recv info: {grad_recv_info}"
        )
        return grad_recv_info

    def _create_grad_send_info(
        self,
        args_recv_info: Tuple,
        kwargs_recv_info: Dict,
    ) -> List[Optional[int]]:
        grad_send_info: List[Optional[int]] = []

        def map_recv_to_send(a):
            if isinstance(a, RecvInfo):
                grad_send_info.append(a.source)
                return a.source
            else:
                grad_send_info.append(None)
                return None

        pippy.fx.node.map_aggregate(args_recv_info, map_recv_to_send)

        pippy.fx.node.map_aggregate(kwargs_recv_info, map_recv_to_send)

        logging.info(
            f"[{self.rank}][{self.name}] " f"Grad send info: {grad_send_info}"
        )
        return grad_send_info

    def _recv_tensor(self, info, recv_reqs):
        logging.debug(
            f"[{self.rank}][{self.name}] "
            f"Receiving tensor '{info.input_name}' from Rank {info.source}: "
            f"{info.buffer.size()}"
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

    def recv_tensor_fn(
        self,
        reqs,
    ):
        return lambda info: self._recv_tensor(info, reqs)

    def _recv_and_fill_inputs(
        self,
        chunk: int,
        args_split,
        kwargs_split,
    ):
        # Receive requests of a chunk
        recv_reqs: List[dist.Work] = []

        act_recv = self.recv_tensor_fn(recv_reqs)

        if args_split:
            chunk_args = args_split[chunk]
            chunk_args_list = list(chunk_args)

        def recv_args(info):
            if isinstance(info, RecvInfo):
                # two RecvInfo of same tensor size
                return act_recv(info)
            else:
                # Y[chunk]
                return chunk_args_list.pop(0)

        composite_args = pippy.fx.node.map_aggregate(
            self.args_recv_info[chunk],
            recv_args,
        )

        if kwargs_split:
            chunk_kwargs = kwargs_split[chunk]

        def recv_kwargs(info):
            if isinstance(info, RecvInfo):
                return act_recv(info)
            else:
                k = next(iter(chunk_kwargs))
                return chunk_kwargs.pop(k)

        composite_kwargs = pippy.fx.node.map_aggregate(
            self.kwargs_recv_info[chunk],
            recv_kwargs,
        )

        # Wait for all recvs to finish
        for work in recv_reqs:
            work.wait()

        return composite_args, composite_kwargs

    def _send_activations(
        self,
        output_tuple,
    ) -> List[dist.Work]:
        # Send requests of a chunk
        send_reqs: List[dist.Work] = []

        for idx, out in enumerate(output_tuple):
            # print(f'[Rank{self.rank}] idx {idx} out {out}')
            dst_ranks = self.act_send_info[idx]
            print(f"[Rank{self.rank}] Destination Ranks: {dst_ranks}")

            for dst in dst_ranks:
                if dst is None:
                    continue
                logging.debug(
                    f"[{self.rank}][{self.name}] "
                    f"Sending tensor to Rank {dst}: {out.size()}"
                )
                work = dist.isend(
                    out,
                    dst
                    if self.group is None
                    else dist.get_global_rank(self.group, dst),  # TODO
                    group=self.group,
                )
                send_reqs.append(work)

        return send_reqs

    def _recv_grads(
        self,
        bwd_chunk,
    ):
        # Receive requests of a chunk
        grad_recv_reqs: List[dist.Work] = []

        recv_grad = self.recv_tensor_fn(grad_recv_reqs)

        # Receive gradients
        grads = pippy.fx.node.map_aggregate(
            self.grad_recv_info[bwd_chunk],
            recv_grad,
        )
        # Wait for all recvs to finish
        for work in grad_recv_reqs:
            work.wait()

        logging.debug(
            f"[{self.rank}][{self.name}] "
            f"Received output grads of chunk {bwd_chunk}: {map_debug_info(grads)}"
        )
        return grads

    def _send_grads(
        self,
        grads_input,
    ) -> List[dist.Work]:
        # Send requests of a chunk
        grad_send_reqs: List[dist.Work] = []

        for grad, grad_receiver in zip(grads_input, self.grad_send_info):
            if isinstance(grad, torch.Tensor) and grad_receiver is not None:
                logging.debug(
                    f"[{self.rank}][{self.name}] "
                    f"Sending gradient to Rank {grad_receiver}: {grad.size()}"
                )
                work = dist.isend(
                    grad,
                    grad_receiver
                    if self.group is None
                    else dist.get_global_rank(
                        self.group, grad_receiver
                    ),  # TODO
                    group=self.group,
                )
                grad_send_reqs.append(work)
            else:
                assert grad is None and grad_receiver is None

        return grad_send_reqs

    def forward_maybe_with_nosync(self, targets, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        else:
            if self.inner_depth > 1:
                if self.rank == self.nstages - 1:
                    for i in range(0, self.inner_depth):
                        if i == 0:
                            out_val = self.submods[i](*args, **kwargs)
                        else:
                            # target_args, target_kwargs = self._recv_and_fill_inputs(
                            # no idea how target is passed in the original code...
                            out_val = self.submods[i](
                                out_val, targets, **kwargs
                            )
                else:
                    for i in range(0, self.inner_depth):
                        if i == 0:
                            out_val = self.submods[i](*args, **kwargs)
                        else:
                            out_val = self.submods[i](out_val, **kwargs)
            else:
                out_val = self.submod(*args, **kwargs)

        return out_val

    def backward_maybe_with_nosync(self, bwd_kwargs: Dict, is_last_chunk: bool):
        if isinstance(self.submod, DistributedDataParallel):
            if is_last_chunk:
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                grads_input, _ = stage_backward(**bwd_kwargs)
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    grads_input, _ = stage_backward(**bwd_kwargs)
        else:
            # Non-DDP submodule, regular backward
            grads_input, _ = stage_backward(**bwd_kwargs)
        return grads_input

    def forward_one_chunk(
        self,
        chunk: int,
        args_split,
        kwargs_split,
        fwd_cache: Dict[int, Any],
    ):
        if self.rank == self.nstages - 1:
            # Need improvement- how do we properly pass targets to forward_maybe_with_nosync?
            targets = args_split[chunk][0]
            print(f"[Rank{self.rank}] args_split {args_split}")
        else:
            targets = None

        composite_args, composite_kwargs = self._recv_and_fill_inputs(
            chunk,
            args_split,
            kwargs_split,
        )

        if self.rank == self.nstages - 1:
            print(f"[Rank{self.rank}] composite_args {composite_args}")

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(
                targets, *composite_args, **composite_kwargs
            )

            print(f"[Rank{self.rank}] Arrived here")


        except Exception as e:
            exc_msg = f"""
            Rank {self.rank} failed to run forward stage: {self.name}
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)
        send_reqs = self._send_activations(output_tuple)

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        fwd_cache[chunk] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        return output, send_reqs

    def backward_one_chunk(
        self,
        bwd_chunk: int,
        fwd_cache: Dict[int, Any],
    ):
        grads = self._recv_grads(bwd_chunk)

        # Pack args for `stage_backward``
        bwd_kwargs = dict(self.bwd_node.kwargs)
        (
            bwd_kwargs["stage_output"],
            bwd_kwargs["input_values"],
        ) = fwd_cache.pop(bwd_chunk)
        # Fill actual gradients received for outputs
        # If nothing received, as in the case of last stage, then we
        # would use the default `output_grads` prepared in the IR phase,
        # i.e. from `bwd_node.kwargs`. For example, it may look like
        # this if there are two outputs: ('None', 'None')
        if len(grads) > 0:
            bwd_kwargs["output_grads"] = grads

        # `stage_backward` node does not have `args`, only `kwargs`
        grads_input = self.backward_maybe_with_nosync(
            bwd_kwargs,
            bwd_chunk == self.chunks - 1,
        )

        grad_send_reqs = self._send_grads(grads_input)
        return grad_send_reqs

    def forward(self, *args, **kwargs):
        # map microbatch ID to list of forward tensor args
        fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}

        args_split = None
        kwargs_split = None
        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self.chunks,
                self.args_chunk_spec,
                self.kwargs_chunk_spec,
            )

        if self.rank == self.nstages - 1:
            # where is my Y?
            print(
                f"[Rank{self.rank}] ArgsSplit: {args_split}, KwargsSplit: {kwargs_split}"
            )

        # Activation send requests of all chunk
        all_send_reqs: List[dist.Work] = []

        output_chunks = [None] * self.chunks

        # Forward pass of all chunks
        for chunk in range(self.chunks):
            s = self.streams[chunk % self.nstreams]
            with torch.cuda.stream(s):
                output, send_reqs = self.forward_one_chunk(
                    chunk, args_split, kwargs_split, fwd_cache
                )
                all_send_reqs += send_reqs
                # Prepare for final output merge or reduction
                output_chunks[chunk] = output

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in all_send_reqs:
            work.wait()

        # Backward starts here
        # Grad send requests of all chunk
        all_grad_send_reqs: List[dist.Work] = []

        for bwd_chunk in range(self.chunks):
            if self.pipe.has_loss_and_backwards:
                grad_send_reqs = self.backward_one_chunk(bwd_chunk, fwd_cache)
                all_grad_send_reqs += grad_send_reqs

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in all_grad_send_reqs:
            work.wait()

        # Last rank return merged results per original format
        if self.rank == self.nstages - 1:
            return merge_chunks(
                output_chunks,
                self.output_chunk_spec,
            )
        else:
            return None


class PipelineStage1F1B(PipelineStage):
    def __init__(
        self,
        pipe: Pipe,
        rank: int,
        nstages: int,
        chunks: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
    ):
        super().__init__(
            pipe,
            rank,
            nstages,
            chunks,
            device,
            group=group,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec,
        )

    def forward(self, *args, **kwargs):
        # map microbatch ID to list of forward tensor args
        fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}

        args_split = None
        kwargs_split = None
        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self.chunks,
                self.args_chunk_spec,
                self.kwargs_chunk_spec,
            )

        # Activation send requests of all chunk
        all_send_reqs: List[dist.Work] = []
        # Grad send requests of all chunk
        all_grad_send_reqs: List[dist.Work] = []
        # Caching chunk outputs for final output merge or reduction
        output_chunks = []

        warmup_chunks = cooldown_chunks = self.nstages

        # Warm-up phase: forward number of chunks equal to pipeline depth.
        for chunk in range(warmup_chunks):
            output, send_reqs = self.forward_one_chunk(
                chunk, args_split, kwargs_split, fwd_cache
            )
            all_send_reqs += send_reqs
            output_chunks.append(output)

        # 1F1B phase
        for bwd_chunk in range(0, self.chunks - cooldown_chunks):
            # Schedule backward for one warmed up chunk
            if self.pipe.has_loss_and_backwards:
                grad_send_reqs = self.backward_one_chunk(bwd_chunk, fwd_cache)
                all_grad_send_reqs += grad_send_reqs

            # Schedule forward for one new chunk
            fwd_chunk = bwd_chunk + warmup_chunks
            output, send_reqs = self.forward_one_chunk(
                fwd_chunk, args_split, kwargs_split, fwd_cache
            )
            all_send_reqs += send_reqs
            # Prepare for final output merge or reduction
            output_chunks.append(output)

        # Cool-down phase: backward for the rest of the chunks
        for bwd_chunk in range(self.chunks - cooldown_chunks, self.chunks):
            if self.pipe.has_loss_and_backwards:
                grad_send_reqs = self.backward_one_chunk(bwd_chunk, fwd_cache)
                all_grad_send_reqs += grad_send_reqs

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in all_send_reqs:
            work.wait()

        for work in all_grad_send_reqs:
            work.wait()

        # Last rank return merged results per original format
        if self.rank == self.nstages - 1:
            return merge_chunks(
                output_chunks,
                self.output_chunk_spec,
            )
        else:
            return None
