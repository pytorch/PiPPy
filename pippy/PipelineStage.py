# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.fx as fx
from torch.fx.node import map_aggregate, map_arg
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn.parallel import DistributedDataParallel

from pippy.backward import stage_backward
from pippy.debug import map_debug_info
from pippy.IR import Pipe
from pippy.microbatch import merge_chunks, split_args_kwargs_into_chunks
from pippy.utils import flatten_args, modify_graph_op_device


logger = logging.getLogger(__name__)


def _make_tensor_from_meta(
    example_value: FakeTensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        example_value.size(), dtype=example_value.dtype, device=device
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
        return f"RecvInfo(input={self.input_name}, source={self.source}, shape={self.buffer.size()})"


class StageArgPlaceholder:
    pass


class StageKwargPlaceholder:
    pass


class PipelineStage(torch.nn.Module):
    def __init__(
        self,
        pipe: Pipe,
        stage_index: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
    ):
        super().__init__()
        self.pipe = pipe
        self.stage_index = stage_index
        self.nstages = pipe.num_stages
        self.chunks = pipe.num_chunks
        self.device = device
        self.group = group
        if dist.get_world_size(self.group) > self.nstages:
            raise RuntimeError(
                "Number of ranks is larger than number of stages, some ranks are unused"
            )

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(group)

        # Run time states
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}
        # Split input chunks
        self.args_split = None
        self.kwargs_split = None
        # Activation send requests of all chunk
        self.all_act_send_reqs: List[dist.Work] = []
        # Grad send requests of all chunk
        self.all_grad_send_reqs: List[dist.Work] = []
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: List[Any] = []

        # Find my submodule
        self.split_gm = self.pipe.split_gm
        named_children = list(self.split_gm.named_children())
        self.name, self.submod = named_children[stage_index]
        logger.info(
            f"[{self.group_rank}] "
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

        # Find my backward node in graph
        if self.pipe.has_loss_and_backwards:
            found_bwd = False
            seen_bwd = -1
            for node in reversed(self.split_gm.graph.nodes):
                if (node.op, node.target) == ("call_function", stage_backward):
                    seen_bwd += 1
                    if seen_bwd == self.stage_index:
                        found_bwd = True
                        self.bwd_node = node
                        break
            if not found_bwd:
                raise AssertionError(
                    f"Cannot find backward for {self.name} in graph"
                )

        # Create submod to rank mapping
        self.submod_to_stage_index: Dict[str, int] = {}
        for i, (name, _) in enumerate(self.split_gm.named_children()):
            self.submod_to_stage_index.setdefault(name, i)

        # Create stage id to group rank mapping
        # In interleaved case, `group_rank` is stage index % group size.
        self.stage_index_to_group_rank: Dict[int, int] = {}
        pg_world_size = dist.get_world_size(group)
        for i in range(self.nstages):
            # We only support wrapped-around interleaving
            peer_rank = i % pg_world_size
            self.stage_index_to_group_rank.setdefault(i, peer_rank)

        # Prepare send/recv infrastructure
        self._prepare_send_recv_infra()
        # Cast submodule to device
        self._move_submod_to_device()
        # Move ops argument to device
        self._move_ops_to_device()

    def _move_submod_to_device(self):
        # Move submodule to indicated device if possible
        # Note: we cannot move meta module to real devices because meta tensors
        # do not support to() method. One needs to do an in-place tensor swap in
        # that case.
        has_meta_param = any(
            isinstance(p, FakeTensor) or p.is_meta
            for p in self.submod.parameters()
        )
        if has_meta_param:
            logger.debug(f"[{self.group_rank}] Found meta parameters!")
        else:
            logger.debug(f"[{self.group_rank}] No meta parameters found!")
            self.submod.to(self.device)

    def _move_ops_to_device(self):
        # Today PT2 tracer does not treat `x.device` as a symbolic device;
        # instead, the device of tracing time got burned into the generated
        # code.  Here we provide a workaround for users to manually modify the
        # "device" kwarg of operations. Such operation may include:
        # `torch.ones`, `torch.zeros`, `torch.rand`, etc.
        modify_graph_op_device(self.submod, self.device)

    def is_first(self):
        return self.stage_index == 0

    def is_last(self):
        return self.stage_index == self.nstages - 1

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

    def get_stage_index_of_submod(
        self,
        submod_name: str,
    ):
        if submod_name not in self.submod_to_stage_index:
            raise AssertionError(f"Stage id of {submod_name} not found")

        return self.submod_to_stage_index[submod_name]

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
                if "example_value" in input_node.meta:
                    real_input_node = input_node.args[0]
                    out_idx = input_node.args[1]
                    return create_recv_tensor(real_input_node, out_idx)
                else:
                    raise NotImplementedError(
                        f"getitem gets a non-Tensor value, this is not yet supported. "
                        f"Node: {input_node.format_node()}"
                    )

            if output_idx is not None:
                # If a node has multiple output values, "example_value" is a list
                # of tensor meta
                example_value = input_node.meta["example_value"][output_idx]
            else:
                example_value = input_node.meta["example_value"]

            logger.info(
                f"[{self.group_rank}] "
                f"Creating recv buffer for input '{input_node.name}' "
                f"value index {output_idx}: {example_value.size()}"
            )

            src_rank = self.get_stage_index_of_submod(input_node.name)
            buffer = _make_tensor_from_meta(example_value, self.device)
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
        args_recv_info = map_arg(self.node.args, create_recv_tensor)

        # `kwargs` is a Dict, hence we will have:
        # Dict[keyword, RecvInfo]
        kwargs_recv_info = map_arg(self.node.kwargs, create_recv_tensor)

        logger.info(
            f"[{self.group_rank}] " f"Activation recv / args info: {args_recv_info}"
        )
        return args_recv_info, kwargs_recv_info

    def find_dst_rank(
        self,
        user: fx.Node,
    ) -> Optional[int]:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        if user.op == "call_module":
            # User is a stage (`call_module`)
            return self.get_stage_index_of_submod(user.name)
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

        for user in self.node.users:
            if user.target is operator.getitem:
                # Recursively find the real destination
                gi_dsts = act_send_info.setdefault(out_idx, [])
                for gi_user in user.users:
                    dst_rank = self.find_dst_rank(gi_user)
                    if dst_rank is not None:
                        gi_dsts.append(dst_rank)
                # Next `getitem` will point to the next output index
                out_idx += 1
            else:
                # In case of single output value, `out_idx` will not increase
                dsts = act_send_info.setdefault(out_idx, [])
                dst_rank = self.find_dst_rank(user)
                if dst_rank is not None:
                    dsts.append(dst_rank)

        logger.info(f"[{self.group_rank}] " f"Send info: {act_send_info}")
        return act_send_info

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Dict[int, RecvInfo]:
        # Dict[output_index, RecvInfo]
        grad_recv_info: Dict = {}
        my_example_value = self.node.meta["example_value"]

        for out_idx, dst_list in act_send_info.items():
            if not dst_list:
                # No actual receiver for activation so no grad coming back
                continue

            # TODO: clean way
            if len(act_send_info) > 1:
                example_value = my_example_value[out_idx]
            else:
                example_value = my_example_value

            # TODO: otherwise needs grad accumulation
            assert len(dst_list) == 1
            grad_src = dst_list[0]
            grad_recv_info[out_idx] = RecvInfo(
                f"{grad_src}",
                grad_src,
                _make_tensor_from_meta(example_value, self.device),
            )

        logger.info(f"[{self.group_rank}] " f"Grad recv info: {grad_recv_info}")
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

        map_aggregate(args_recv_info, map_recv_to_send)

        map_aggregate(kwargs_recv_info, map_recv_to_send)

        logger.info(f"[{self.group_rank}] " f"Grad send info: {grad_send_info}")
        return grad_send_info

    def _recv_tensor(self, info, recv_reqs):
        logger.debug(
            f"[{self.group_rank}] "
            f"Receiving tensor '{info.input_name}' from Rank {info.source}: "
            f"{info.buffer.size()}"
        )
        # Use async to parallelize recv of tensors
        peer_rank = self.stage_index_to_group_rank[info.source]
        work = dist.irecv(
            info.buffer,
            peer_rank
            if self.group is None
            else dist.get_global_rank(self.group, peer_rank),
            group=self.group,
        )
        recv_reqs.append(work)
        return info.buffer

    def recv_tensor_fn(
        self,
        reqs,
    ):
        return lambda info: self._recv_tensor(info, reqs)

    def split_inputs(self, args, kwargs):
        self.args_split = None
        self.kwargs_split = None
        if args or kwargs:
            self.args_split, self.kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self.chunks,
                self.pipe.args_chunk_spec,
                self.pipe.kwargs_chunk_spec,
            )

    def _recv_and_fill_inputs(
        self,
        chunk: int,
    ):
        # Receive requests of a chunk
        recv_reqs: List[dist.Work] = []

        act_recv = self.recv_tensor_fn(recv_reqs)

        if self.args_split:
            chunk_args = self.args_split[chunk]
            chunk_args_list = list(chunk_args)

        def recv_args(info):
            if isinstance(info, RecvInfo):
                # This is an activation to receive
                return act_recv(info)
            else:
                # This is a pass-in argument
                if len(chunk_args_list):
                    return chunk_args_list.pop(0)  # type: ignore[has-type]
                else:
                    # kwargs were treated as args in graph phase. That's why
                    # there are extra placeholders here. We mark them and filter
                    # them out later.
                    return StageKwargPlaceholder()

        composite_args = map_aggregate(
            self.args_recv_info[chunk],
            recv_args,
        )
        # Filter out kwarg placeholders
        composite_args = tuple(x for x in composite_args if not isinstance(x, StageKwargPlaceholder))

        # Middle stages won't have incoming activations in kwargs form. So if
        # kwargs_split is not empty, it must be model inputs for stage 0. We
        # hence pass it as is to the interal submodule, without performing
        # `recv_args` on it.
        if self.kwargs_split:
            composite_kwargs = self.kwargs_split[chunk]
        else:
            composite_kwargs = {}

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
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                logger.debug(
                    f"[{self.group_rank}] "
                    f"Sending tensor to Rank {dst}: {out.size()}"
                )
                peer_rank = self.stage_index_to_group_rank[dst]
                work = dist.isend(
                    out,
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank),  # TODO
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
        grads = map_aggregate(
            self.grad_recv_info[bwd_chunk],
            recv_grad,
        )
        # Wait for all recvs to finish
        for work in grad_recv_reqs:
            work.wait()

        logger.debug(
            f"[{self.group_rank}] "
            f"Received output grads of chunk {bwd_chunk}: {map_debug_info(grads)}"
        )
        return grads

    def _send_grads(
        self,
        grads_input,
    ) -> List[dist.Work]:
        # Send requests of a chunk
        grad_send_reqs: List[dist.Work] = []

        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                logger.debug(
                    f"[{self.group_rank}] "
                    f"Sending gradient to Rank {grad_recv_stage}: {grad.size()}"
                )
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                work = dist.isend(
                    grad,
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank),  # TODO
                    group=self.group,
                )
                grad_send_reqs.append(work)
            else:
                assert grad is None and grad_recv_stage is None

        return grad_send_reqs

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
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
    ):
        composite_args, composite_kwargs = self._recv_and_fill_inputs(chunk)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(
                *composite_args, **composite_kwargs
            )

        except Exception as e:
            exc_msg = f"""
            Rank {self.group_rank} failed to run forward stage: {self.name}
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        if type(output) is list:
            # HACK: this is a hacky workaround for the fact that export creates
            # output in list format
            output = tuple(output)
        logger.debug(map_debug_info(output))
        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)
        # Prepare for final output merge or reduction
        self.output_chunks.append(output)

        # Send activations
        send_reqs = self._send_activations(output_tuple)
        self.all_act_send_reqs += send_reqs

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[chunk] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

    def backward_one_chunk(
        self,
        bwd_chunk: int,
    ):
        if not self.pipe.has_loss_and_backwards:
            return None

        grads = self._recv_grads(bwd_chunk)

        # Pack args for `stage_backward``
        bwd_kwargs = dict(self.bwd_node.kwargs)
        (
            bwd_kwargs["stage_output"],
            bwd_kwargs["input_values"],
        ) = self.fwd_cache.pop(bwd_chunk)
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
        self.all_grad_send_reqs += grad_send_reqs

    def clear_runtime_states(self):
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Activation send requests of all chunk
        self.all_act_send_reqs.clear()
        # Grad send requests of all chunk
        self.all_grad_send_reqs.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()

    def merge_output_chunks(self):
        return merge_chunks(
            self.output_chunks,
            self.pipe.output_chunk_spec,
        )

    def forward(self, *args, **kwargs):
        # Clean per iteration
        self.clear_runtime_states()

        # Split inputs into chunks
        self.split_inputs(args, kwargs)

        # Forward pass of all chunks
        for chunk in range(self.chunks):
            self.forward_one_chunk(chunk)
            logger.debug(f"[{self.group_rank}] Forwarded chunk {chunk}")

        # Backward starts here

        for bwd_chunk in range(self.chunks):
            self.backward_one_chunk(bwd_chunk)
            logger.debug(f"[{self.group_rank}] Backwarded chunk {bwd_chunk}")

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_act_send_reqs:
            work.wait()

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_grad_send_reqs:
            work.wait()

        # Last rank return merged results per original format
        if self.is_last():
            return self.merge_output_chunks()
        else:
            return None


class PipelineStage1F1B(PipelineStage):
    def __init__(
        self,
        pipe: Pipe,
        rank: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
    ):
        super().__init__(
            pipe,
            rank,
            device,
            group=group,
        )

    def forward(self, *args, **kwargs):
        # Clean per iteration
        self.clear_runtime_states()

        # Split inputs into chunks
        self.split_inputs(args, kwargs)

        warmup_chunks = cooldown_chunks = self.nstages

        # Warm-up phase: forward number of chunks equal to pipeline depth.
        for chunk in range(warmup_chunks):
            self.forward_one_chunk(chunk)

        # 1F1B phase
        for bwd_chunk in range(0, self.chunks - cooldown_chunks):
            # Schedule backward for one warmed up chunk
            self.backward_one_chunk(bwd_chunk)

            # Schedule forward for one new chunk
            fwd_chunk = bwd_chunk + warmup_chunks
            self.forward_one_chunk(fwd_chunk)

        # Cool-down phase: backward for the rest of the chunks
        for bwd_chunk in range(self.chunks - cooldown_chunks, self.chunks):
            self.backward_one_chunk(bwd_chunk)

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_act_send_reqs:
            work.wait()

        for work in self.all_grad_send_reqs:
            work.wait()

        # Last rank return merged results per original format
        if self.is_last():
            return self.merge_output_chunks()
        else:
            return None
