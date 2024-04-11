# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_aggregate
from torch.nn.parallel import DistributedDataParallel

from pippy.backward import stage_backward
from pippy.debug import map_debug_info
from pippy.IR import Pipe
from pippy.utils import flatten_args, modify_graph_op_device

logger = logging.getLogger(__name__)


class PipelineStageBase(ABC):
    def __init__(
        self,
        submodule: torch.nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        num_microbatches: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        if stage_index >= num_stages:
            raise ValueError(
                f"Stage index {stage_index} is out of range of {num_stages}"
            )

        self.submod = submodule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.chunks = num_microbatches
        self.group = group

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.group_size} cannot be larger than number of stages {self.num_stages}"
            )

        # Run time states
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: Dict[int, Tuple[Any, List[torch.Tensor]]] = {}
        # Current forward chunk id
        self.fwd_chunk_id: int = 0
        # Current backward chunk id
        self.bwd_chunk_id: int = 0
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: List[Any] = []

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
        # Reset pointers
        self.fwd_chunk_id = 0
        self.bwd_chunk_id = 0
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()

    def _retrieve_recv_activations(
        self,
    ):
        """
        Retrieve the activations received for the current stage during forward.
        """
        raise NotImplementedError

    def _retrieve_recv_grads(
        self,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        """
        raise NotImplementedError

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        else:
            out_val = self.submod(*args, **kwargs)
        return out_val

    def backward_maybe_with_nosync(self, bwd_kwargs: Dict, bwd_chunk_id: int):
        if isinstance(self.submod, DistributedDataParallel):
            if bwd_chunk_id == self.chunks - 1:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                grads_input = stage_backward(**bwd_kwargs)
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    grads_input = stage_backward(**bwd_kwargs)
        else:
            # Non-DDP submodule, regular backward
            grads_input = stage_backward(**bwd_kwargs)

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
            exc_msg = f"""s
            Stage {self.stage_index} failed to run forward:
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
            f"[{self.stage_index}] Forwarded chunk {self.fwd_chunk_id}, outputs: {map_debug_info(output)}"
        )
        self.fwd_chunk_id += 1
        return output

    def backward_one_chunk(
        self,
        loss=None,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.
        """
        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(self.bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads()
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        self.grads_input = self.backward_maybe_with_nosync(
            bwd_kwargs, self.bwd_chunk_id
        )
        logger.debug(
            f"[{self.stage_index}] Backwarded chunk {self.bwd_chunk_id}"
        )
        self.bwd_chunk_id += 1


def _make_tensor_from_meta(
    example: FakeTensor,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(
        example.size(),
        dtype=example.dtype,
        layout=example.layout,
        device=device,
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


class RootArgPlaceholder:
    pass


# An input can be either a received activation or a model input
InputInfo = Union[RecvInfo, RootArgPlaceholder]


class _PipelineStage(PipelineStageBase):
    def __init__(
        self,
        stage_module: torch.nn.Module,
        stage_index: int,
        pipe_info: Pipe.PipeInfo,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
    ):
        PipelineStageBase.__init__(
            self,
            stage_module,
            stage_index,
            pipe_info.num_stages,
            device,
            pipe_info.num_chunks,
            group,
        )
        self.pipe_info = pipe_info

        # Find stage nodes in graph
        submod_nodes = [
            node for node in pipe_info.graph.nodes if node.op == "call_module"
        ]
        if len(submod_nodes) != self.num_stages:
            raise AssertionError(
                f"Number of submodules in pipe graph {len(submod_nodes)} does not match number of stages {self.num_stages}"
            )

        # Find my stage node in graph
        self.node = submod_nodes[self.stage_index]
        self.name = self.node.name
        logger.info(
            f"[{self.group_rank}] "
            f"Creating PipelineStage {stage_index} for {self.name}"
        )

        # Create mapping from stage name to stage index
        self.submod_to_stage_index: Dict[str, int] = {}
        for i, node in enumerate(submod_nodes):
            self.submod_to_stage_index.setdefault(node.name, i)

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
            logger.debug(f"[{self.stage_index}] Found meta parameters!")
        else:
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
        def create_recv_tensor(placeholder, arg_node):
            """
            Create a receive buffer for a placeholder.
            """
            if arg_node.op == "placeholder":
                # This is a root level placeholder, thus an input argument to the entire model.
                # We are likely at stage 0, hence no need to create a receive buffer.
                return RootArgPlaceholder()

            # Figure out the source stage of this input
            while arg_node.target is operator.getitem:
                # If the input is a getitem, we need to go deeper
                arg_node = arg_node.args[0]

            assert (
                arg_node.op == "call_module"
            ), f"Expecting call_module, got {arg_node.op}"
            src_stage = self.get_stage_index_of_submod(arg_node.name)

            # Create a receive buffer for this placeholder
            example_value = placeholder.meta["val"]
            logger.info(
                f"[{self.stage_index}] "
                f"Creating recv buffer for input '{placeholder.name}' "
                f": {example_value.shape}, {example_value.dtype}"
            )
            buffer = _make_tensor_from_meta(example_value, self.device)

            return RecvInfo(
                arg_node.name,
                src_stage,
                buffer,
            )

        args_recv_info: List[InputInfo] = []
        # Filter out placeholder nodes from `self.submod` (a GraphModule)
        placeholders = filter(
            lambda node: node.op == "placeholder", self.submod.graph.nodes
        )
        # `placeholders` are nodes internal to submod.
        # `self.node.args` are dependency nodes in the outer graph.
        # The two are 1:1.
        for placeholder, arg_node in zip(placeholders, self.node.args):
            # Create a receive buffer for this placeholder
            recv_info = create_recv_tensor(placeholder, arg_node)
            args_recv_info.append(recv_info)

        logger.info(
            f"[{self.stage_index}] "
            f"Activation recv / args info: {args_recv_info}"
        )
        # `args` is a Tuple, hence we will return a Tuple[InputInfo]
        return tuple(args_recv_info)

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

        logger.info(f"[{self.stage_index}] " f"Send info: {act_send_info}")
        return act_send_info

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Tuple[RecvInfo, ...]:
        # Dict[output_index, RecvInfo]
        grad_recv_info: Dict[int, RecvInfo] = {}
        output_nodes = [
            node for node in self.submod.graph.nodes if node.op == "output"
        ]
        assert len(output_nodes) == 1
        output_node = output_nodes[0]
        # The output node may take multiple args, meaning the submod having multiple output values.
        output_vals = flatten_args(output_node.args)

        for out_idx, dst_list in act_send_info.items():
            if not dst_list:
                # No actual receiver for activation so no grad coming back
                continue

            output = output_vals[out_idx]
            example_value = output.meta["val"]
            logger.debug(
                f"[{self.stage_index}] Creating grad recv buffer for output {output.name} "
                f": {example_value.shape}, {example_value.dtype}"
            )

            # TODO: otherwise needs grad accumulation
            assert (
                len(dst_list) == 1
            ), "Backward of skip connections not supported yet"
            grad_src = dst_list[0]
            grad_recv_info[out_idx] = RecvInfo(
                f"{grad_src}",
                grad_src,
                _make_tensor_from_meta(example_value, self.device),
            )

        # Convert to tuple for convenience in get_ops and retrieve tensor
        grad_recv_info_tuple = tuple(grad_recv_info.values())
        logger.info(
            f"[{self.stage_index}] " f"Grad recv info: {grad_recv_info_tuple}"
        )
        return grad_recv_info_tuple

    def _create_grad_send_info(
        self,
        args_recv_info: Tuple,
    ) -> List[Optional[int]]:
        grad_send_info: List[Optional[int]] = []

        def map_recv_to_send(a):
            # Note: we send gradients back to previous stage as long as in
            # forward it is a received input, regardless of whether it requires
            # grad. It is up to the previous stage to disgard this gradient.
            if isinstance(a, RecvInfo):
                grad_send_info.append(a.source)
                return a.source
            else:
                grad_send_info.append(None)
                return None

        map_aggregate(args_recv_info, map_recv_to_send)

        logger.info(
            f"[{self.stage_index}] " f"Grad send info: {grad_send_info}"
        )
        return grad_send_info

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
                    f"[{self.stage_index}] "
                    f"Sending tensor to Stage {dst}: {out.size()}"
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
                    f"[{self.stage_index}] "
                    f"Sending gradient to Stage {grad_recv_stage}: {grad.size()}"
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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "forward() function of PipelineStage is deprecated; please create a "
            "PipelineSchedule and call step(input) instead."
        )


# Backward compatibility support for creation out of Pipe
class PipelineStage(_PipelineStage):
    def __init__(
        self,
        pipe: Pipe,
        stage_index: int,
        device: torch.device,
        group: dist.ProcessGroup = None,
    ):
        # Find my stage module
        stage_module = pipe.get_stage_module(stage_index)
        # Get my pipe info
        pipe_info = pipe.info()
        super().__init__(stage_module, stage_index, pipe_info, device, group)
