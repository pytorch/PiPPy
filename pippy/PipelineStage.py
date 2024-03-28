# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_aggregate, map_arg
from torch.nn.parallel import DistributedDataParallel

from pippy.backward import stage_backward
from pippy.debug import map_debug_info
from pippy.IR import Pipe
from pippy.microbatch import merge_chunks, split_args_kwargs_into_chunks
from pippy.utils import flatten_args, modify_graph_op_device, QualnameMapMixin

logger = logging.getLogger(__name__)


class PipelineStageBase(ABC):
    def __init__(
        self,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
    ):
        super().__init__()
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group

        # TODO: rename `rank` to `group_rank`
        self.rank = dist.get_rank(self.group)

        # TODO: rename `world_size`` to `group_size`
        self.world_size = dist.get_world_size(self.group)
        if self.world_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.world_size} cannot be larger than number of stages {self.num_stages}"
            )

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        self.has_backward = False

    @property
    def has_backward(self) -> bool:
        return self._has_backward

    @has_backward.setter
    def has_backward(self, has_backward: bool):
        self._has_backward = has_backward

    @property
    def is_first(self):
        return self.stage_index == 0

    @property
    def is_last(self):
        return self.stage_index == self.num_stages - 1

    @abstractmethod
    def forward_one_chunk(self, *args, **kwargs):
        """
        Perform forward pass on the module.
        This should only be called once per microbatch.

        Args:
            microbatch: The input to the module
        """
        raise NotImplementedError

    @abstractmethod
    def get_fwd_recv_ops(self) -> List[dist.P2POp]:
        """
        Get the list of P2P operations that need to be performed before calling forward()
        """
        raise NotImplementedError

    @abstractmethod
    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        """
        Get the list of P2P operations that need to be performed after calling forward()
        """
        raise NotImplementedError

    @abstractmethod
    def backward_one_chunk(self, **kwargs):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.
        """
        raise NotImplementedError

    @abstractmethod
    def get_bwd_recv_ops(self) -> List[dist.P2POp]:
        """
        Get the list of P2P operations that need to be performed before calling backward()
        """
        raise NotImplementedError

    @abstractmethod
    def get_bwd_send_ops(self) -> List[dist.P2POp]:
        """
        Get the list of P2P operations that need to be performed after calling backward()
        """
        raise NotImplementedError

    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
        raise NotImplementedError

    def split_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        # TODO: lift an implementation here from PipelineStage or move it to PipelineSchedule
        raise NotImplementedError

    def merge_outputs(self):
        """
        Merges the outputs of the microbatches into a single output.
        """
        # TODO: lift an implementation here from PipelineStage or move it to PipelineSchedule
        raise NotImplementedError


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


# An input can be either a received activation or a model input
InputInfo = Union[RecvInfo, StageArgPlaceholder]


class PipelineStage(PipelineStageBase, QualnameMapMixin):
    def __init__(
        self,
        pipe: Pipe,
        stage_index: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
    ):
        PipelineStageBase.__init__(
            self, stage_index, pipe.num_stages, device, group
        )
        self.pipe = pipe
        self.chunks = pipe.num_chunks

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(group)

        # Run time states
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}
        # Current forward chunk id
        self.fwd_chunk_id: int = 0
        # Current backward chunk id
        self.bwd_chunk_id: int = 0
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

        # Enable `remap_qualname` method
        QualnameMapMixin.__init__(
            self,
            pipe.submod_qualname_mappings[self.name],
            pipe.tracer_qualname_map,
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

        # Create submod to rank mapping
        self.submod_to_stage_index: Dict[str, int] = {}
        for i, (name, _) in enumerate(self.split_gm.named_children()):
            self.submod_to_stage_index.setdefault(name, i)

        # Create stage id to group rank mapping
        # In interleaved case, `group_rank` is stage index % group size.
        self.stage_index_to_group_rank: Dict[int, int] = {}
        pg_world_size = dist.get_world_size(group)
        for i in range(self.num_stages):
            # We only support wrapped-around interleaving
            peer_rank = i % pg_world_size
            self.stage_index_to_group_rank.setdefault(i, peer_rank)

        # `has_backward` will be set by PipelineSchedule
        self.has_backward = False

        # Prepare forward send/recv infrastructure
        self._prepare_forward_infra()
        # Backward infra will created lazily
        self.grad_recv_info: Dict = {}
        self.grad_send_info: Optional[List] = None

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

    def _prepare_forward_infra(self):
        """
        Create send/recv infrastructures for activations (during forward)
        """
        # chunk : Tuple of arg buffers
        self.args_recv_info: Dict[int, Tuple[InputInfo]] = {}
        # Flag per chunk to keep track of whether we have set `requires_grad`
        # for receive buffers. Format: {chunk : Boolean}
        self.set_requires_grad: Dict[int, bool] = {}
        for chunk in range(self.chunks):
            self.args_recv_info[chunk] = self._create_act_recv_buffers()
            self.set_requires_grad[chunk] = False

        # Send info during forward for each activation
        self.act_send_info = self._create_act_send_info()

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
            return RecvInfo(
                input_node.name,
                src_rank,
                buffer,
            )

        # `args` is a Tuple, hence we will have:
        # Tuple[InputInfo]
        args_recv_info = map_arg(self.node.args, create_recv_tensor)

        logger.info(
            f"[{self.group_rank}] "
            f"Activation recv / args info: {args_recv_info}"
        )
        return args_recv_info

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
    ) -> Tuple[RecvInfo, ...]:
        # Dict[output_index, RecvInfo]
        grad_recv_info: Dict[int, RecvInfo] = {}
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

        # Convert to tuple for convenience in get_ops and retrieve tensor
        grad_recv_info_tuple = tuple(grad_recv_info.values())
        logger.info(
            f"[{self.group_rank}] " f"Grad recv info: {grad_recv_info_tuple}"
        )
        return grad_recv_info_tuple

    def _create_grad_send_info(
        self,
        args_recv_info: Tuple,
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

        logger.info(f"[{self.group_rank}] " f"Grad send info: {grad_send_info}")
        return grad_send_info

    def split_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self.chunks,
                self.pipe.args_chunk_spec,
                self.pipe.kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self.chunks, [{}] * self.chunks

    def _get_recv_ops(
        self,
        recv_infos: Tuple[InputInfo],
    ) -> List[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: List[dist.P2POp] = []
        for info in recv_infos:
            if not isinstance(info, RecvInfo):
                continue

            peer_rank = self.stage_index_to_group_rank[info.source]
            peer_global_rank = (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )  # TODO
            ops.append(
                dist.P2POp(
                    dist.irecv, info.buffer, peer_global_rank, self.group
                )
            )

        return ops

    def get_fwd_recv_ops(self) -> List[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: Tuple[InputInfo] = self.args_recv_info[self.fwd_chunk_id]

        # In case there is backward pass, set requires_grad for receive buffers
        # before first forward
        if self.has_backward and not self.set_requires_grad[self.fwd_chunk_id]:
            for a in recv_infos:
                if isinstance(a, RecvInfo):
                    a.buffer.requires_grad_(True)

        return self._get_recv_ops(recv_infos)

    def get_bwd_recv_ops(self) -> List[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        # Create bwd recv infra lazily
        recv_infos = self.grad_recv_info.setdefault(
            self.bwd_chunk_id,
            # `grad_recv_info` is a mirror of `act_send_info`
            self._create_grad_recv_info(self.act_send_info),
        )

        return self._get_recv_ops(recv_infos)

    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
        # Use "-1" to get the outputs created by the last chunk
        output = self.output_chunks[-1]
        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        ops: List[dist.P2POp] = []

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
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )  # TODO
                ops.append(
                    dist.P2POp(dist.isend, out, peer_global_rank, self.group)
                )

        return ops

    def get_bwd_send_ops(self) -> List[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        """
        if not self.has_backward or self.is_first:
            return []

        # Create bwd send infra lazily
        if self.grad_send_info is None:
            # Send info for input grads during backward:
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info`
            self.grad_send_info = self._create_grad_send_info(
                self.args_recv_info[0]
            )

        ops: List[dist.P2POp] = []
        for grad, grad_recv_stage in zip(self.grads_input, self.grad_send_info):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                logger.debug(
                    f"[{self.group_rank}] "
                    f"Sending gradient to Rank {grad_recv_stage}: {grad.size()}"
                )
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )  # TODO
                ops.append(
                    dist.P2POp(dist.isend, grad, peer_global_rank, self.group)
                )
            else:
                assert grad is None and grad_recv_stage is None

        return ops

    def _map_tensor_from_recv_info(
        self,
        recv_infos: Tuple[InputInfo],
    ):
        def get_recv_tensor(info):
            if isinstance(info, RecvInfo):
                return info.buffer
            else:
                raise AssertionError(f"Expected RecvInfo but got {type(info)}")

        tensors = map_aggregate(
            recv_infos,
            get_recv_tensor,
        )

        return tensors

    def _retrieve_recv_activations(
        self,
    ):
        """
        Retrieve the activations received for the current stage during forward.
        """
        recv_infos = self.args_recv_info[self.fwd_chunk_id]
        activations = self._map_tensor_from_recv_info(recv_infos)
        return activations

    def _retrieve_recv_grads(
        self,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        """
        recv_infos = self.grad_recv_info[self.bwd_chunk_id]
        grads = self._map_tensor_from_recv_info(recv_infos)
        return grads

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
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
            composite_kwargs = kwargs or {}
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = self._retrieve_recv_activations()
            composite_kwargs = {}

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

        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)
        # Prepare for final output merge or reduction
        self.output_chunks.append(output)

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[self.fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            f"[{self.group_rank}] Forwarded chunk {self.fwd_chunk_id}, outputs: {map_debug_info(output)}"
        )
        self.fwd_chunk_id += 1
        return output

    def backward_one_chunk(
        self,
        loss=None,
    ):
        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(self.bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            self.grads_input = torch.autograd.grad(
                loss,
                input_values,
                allow_unused=True,  # TODO
            )
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads()

            if self.is_first:
                # If an input to the pipeline requires gradient,
                # `torch.autograd.backward` will accumulate the gradient into the
                # `.grad` field of such input
                torch.autograd.backward(
                    stage_output,
                    grad_tensors=grads_output,
                )
            else:
                # Use `torch.autograd.grad` to return the gradients
                self.grads_input = torch.autograd.grad(
                    stage_output,
                    input_values,
                    grads_output,
                    allow_unused=True,  # TODO
                )

        logger.debug(
            f"[{self.group_rank}] Backwarded chunk {self.bwd_chunk_id}"
        )
        self.bwd_chunk_id += 1

    def clear_runtime_states(self):
        # Reset pointers
        self.fwd_chunk_id = 0
        self.bwd_chunk_id = 0
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()

    def merge_outputs(self):
        # Last rank return merged results per original format
        if self.is_last:
            return merge_chunks(
                self.output_chunks,
                self.pipe.output_chunk_spec,
            )
        else:
            return None

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "forward() function of PipelineStage is deprecated; please create a "
            "PipelineSchedule and call step(input) instead."
        )
