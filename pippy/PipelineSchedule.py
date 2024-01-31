# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class PipelineStage(ABC, nn.Module):
    @abstractmethod
    def forward(self, args: List[torch.tensor]) -> torch.tensor:
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
    def backward(self):
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

    @abstractmethod
    def compute_loss(self):
        """
        Compute loss from the outputs of the last stage
        """
        raise NotImplementedError


class PipelineStageV2Impl(PipelineStage):
    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        rank: int,
        world_size: int,
        inputs_meta: List[torch.tensor],
        outputs_meta: Optional[List[torch.tensor]] = None,
    ):
        super().__init__()
        self.module = module
        self.rank = rank
        self.stage_id = stage_id
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1
        self.num_stages = num_stages

        if outputs_meta is None:
            outputs_meta = module.forward(*inputs_meta)

        self.fwd_inputs: List[torch.tensor] = self.create_buffers(inputs_meta)
        self.fwd_outputs = None
        self.fwd_output_grads: List[torch.tensor] = self.create_buffers(
            outputs_meta
        )
        self.fwd_outputs_for_backward: Deque[
            Tuple[torch.tensor, torch.tensor]
        ] = deque()
        logger.debug(
            f"""
            {[fwd_input.shape for fwd_input in self.fwd_inputs if isinstance(fwd_input, torch.Tensor)]},
            {[fwd_output_grad.shape for fwd_output_grad in self.fwd_output_grads]}
            """
        )

        self.prev_stage = (rank - 1) % world_size
        self.next_stage = (rank + 1) % world_size

        self.fwd_recv_queue = None
        self.bwd_recv_queue = None

        self.requests: List[dist.P2POp] = []
        logger.debug(
            f"""
            finished pipeline stage init, {self.stage_id=}, {self.is_first_stage=},
            {self.is_last_stage=}, {self.num_stages=},
            """
        )

    def to(self, *args, **kwargs):
        """
        Move the module to a new device or data type, including the buffers.
        """
        super().to(*args, **kwargs)

        # find the device of the underlying module and move the buffers to it if they are meta
        device = next(self.module.parameters()).device

        for i, fwd_input in enumerate(self.fwd_inputs):
            if fwd_input.is_meta:
                self.fwd_inputs[i] = torch.empty_like(fwd_input, device=device)
            self.fwd_inputs[i] = self.fwd_inputs[i].to(*args, **kwargs)
        for i, fwd_output_grad in enumerate(self.fwd_output_grads):
            if fwd_output_grad.is_meta:
                self.fwd_output_grads[i] = torch.empty_like(
                    fwd_output_grad, device=device
                )
            self.fwd_output_grads[i] = self.fwd_output_grads[i].to(
                *args, **kwargs
            )

    def create_buffers(
        self, inputs_meta: List[torch.tensor]
    ) -> List[torch.Tensor]:
        """
        Creates buffers for a given input on a specified device.
        This function takes as input a meta tensor or a list of meta tensors
        and returns a flattened list of empty tensors of the same shape.
        """
        if isinstance(inputs_meta, torch.Tensor):
            return [inputs_meta]
        elif isinstance(inputs_meta, (list, tuple)):
            return [
                item
                for sublist in [self.create_buffers(inp) for inp in inputs_meta]
                for item in sublist
            ]
        raise ValueError(
            f"Unsupported input type {type(inputs_meta)} cannot create buffers"
        )

    def init_p2p_neighbors(self):
        """
        Set up p2p communitors between previous and next stages
        by sending a dummy tensor.

        If this is used, must be called for all pipeline stages.
        """
        ops = []
        recv_tensor = torch.zeros(1, device="cuda")
        send_tensor = torch.ones(1, device="cuda")
        # forward
        if not self.is_first_stage:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.prev_stage))
        if not self.is_last_stage:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.next_stage))

        # backward
        if not self.is_first_stage:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.prev_stage))
        if not self.is_last_stage:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.next_stage))

        return True

    def get_fwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        return [
            dist.P2POp(dist.irecv, fwd_input, self.prev_stage)
            for fwd_input in self.fwd_inputs
        ]

    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        assert (
            self.fwd_outputs is not None
        ), "forward() must be called before get_fwd_send_ops"
        if self.is_last_stage:
            return []
        return [
            dist.P2POp(dist.isend, fwd_output, self.next_stage)
            for fwd_output in self.fwd_outputs
        ]

    def forward(self, args: List[torch.tensor]) -> torch.tensor:
        logger.debug(f"[{self.rank} FORWARD {self.stage_id}")
        if self.is_first_stage:
            self.fwd_inputs = args

        # this is needed when we access the gradients for this in backward()
        for tensor in self.fwd_inputs:
            tensor.requires_grad = True
            tensor.retain_grad()

        # perform forward pass on module
        self.fwd_outputs = self.module(*self.fwd_inputs)

        output_for_backward = (
            self.compute_loss() if self.is_last_stage else self.fwd_outputs
        )

        # we store a ref to the input/output pair for this forward to be later used by the corresponding backward
        self.fwd_outputs_for_backward.append(
            (self.fwd_inputs, output_for_backward)
        )

        return self.fwd_outputs

    def get_bwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_last_stage:
            return []
        return [
            dist.P2POp(dist.irecv, output_grad, self.next_stage)
            for output_grad in self.fwd_output_grads
        ]

    def get_bwd_send_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        for fwd_input in self.fwd_inputs:
            logger.debug(f"{fwd_input.grad=}")
            assert fwd_input.grad is not None, "grad must be valid"
        return [
            dist.P2POp(dist.isend, fwd_input.grad, self.prev_stage)
            for fwd_input in self.fwd_inputs
        ]

    def backward(self):
        logger.debug(f"[{self.rank} BACKWARD {self.stage_id}]")

        if self.is_last_stage:
            self.fwd_inputs, loss = self.fwd_outputs_for_backward.popleft()
        else:
            (
                self.fwd_inputs,
                fwd_outputs,
            ) = self.fwd_outputs_for_backward.popleft()

        # Compute gradients
        if self.is_last_stage:
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=self.fwd_inputs,
                retain_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )
        else:
            gradients = torch.autograd.grad(
                outputs=fwd_outputs,
                inputs=self.fwd_inputs,
                grad_outputs=self.fwd_output_grads,
                retain_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )

        # Set the gradients for each tensor in self.fwd_inputs
        for i in range(len(self.fwd_inputs)):
            self.fwd_inputs[i].grad = gradients[i]

        return self.fwd_inputs

    def compute_loss(self):
        if self.fwd_outputs is None:
            raise RuntimeError("forward() must be called before compute_loss()")
        # TODO: use a real loss function passed in
        return self.fwd_outputs[0].mean()


class PipelineSchedule(ABC):
    @abstractmethod
    def step(self, microbatches: List[torch.Tensor]) -> None:
        """
        Run one iteration of the pipeline schedule. Will go through all the microbatches
        according to the schedule implementation.

        Args:
            microbatches: list of microbatch tensors
        """
        raise NotImplementedError


class PipelineScheduleGPipe(PipelineSchedule):
    def __init__(self, stage: PipelineStage):
        self._stage = stage

    def step(self, microbatches):
        for i, mb in enumerate(microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                logger.debug(f"{mb=}")
                logger.debug(f"len {len(mb)}")
                self._stage.forward(mb)

                ops = self._stage.get_fwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

                logger.debug(
                    f"{self._stage.stage_id} forward mb {i} finished, microbatch: {[inp.shape for inp in mb]}"
                )

        for i, _ in enumerate(microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                self._stage.backward()

                ops = self._stage.get_bwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

            logger.debug(f"{self._stage.stage_id} backward mb {i} finished")


class PipelineScheduleLoopedBFS(PipelineSchedule):
    def __init__(self, stages: List[PipelineStage]):
        self._stages = stages

    def step(self, microbatches):
        for s, stage in enumerate(self._stages):
            for i, mb in enumerate(microbatches):
                with record_function(f"Stage {s} Forward"):
                    ops = stage.get_fwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    stage.forward(mb)

                    ops = stage.get_fwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)

        for stage in reversed(self._stages):
            for i in range(len(microbatches)):
                with record_function(f"Stage {stage.stage_id} Backward"):
                    ops = stage.get_bwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    stage.backward()

                    ops = stage.get_bwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)


class PipelineScheduleLoopedDFS(PipelineSchedule):
    def __init__(self, stages: List[PipelineStage], n_microbatch, pp_id, n_pp):
        assert (
            n_microbatch % n_pp == 0
        ), f"Looped DFS schedule requires microbatch_size ({n_microbatch}) to be a multiple of n_pp ({n_pp})"

        self.stages = stages
        self.n_microbatch = n_microbatch

        self.n_local_stages = len(stages)
        self.total_stages = self.n_local_stages * n_pp
        # world_size
        self.n_pp = n_pp

        self.stage_id_to_global_stage_id = [
            (i * n_pp) + pp_id for i in range(self.n_local_stages)
        ]

        # pp_id is the same as local rank within the PP dimension
        self.pp_id = pp_id

        # number of sequences (chunks)
        self.seq_size = n_pp

        # warmup steps for latest pp stage is trivial to compute
        # increment warmup_steps by 2 for each hop away
        self.warmup_steps = (len(stages) - 1) * self.seq_size
        self.warmup_steps += 2 * ((n_pp - 1) - pp_id)
        self.forward_steps = len(stages) * n_microbatch
        self.total_steps = self.warmup_steps + (len(stages) * n_microbatch)
        logger.debug(
            f"pp_id {pp_id} warmup_steps {self.warmup_steps} forward_steps {self.forward_steps} total_steps {self.total_steps}"
        )

    def step(self, microbatches):
        """
        # n_loop = n_stage / n_pp
        # run microbatches in sequences of NPp

        schedule operates at the rank level

        highest rank has a warmup (F only) count of [len(stages) - 1] * seq_size
        each hop away from highest rank adds 2 warmup stages
            - one happened before highest rank's warmup started,
            - one waiting for backward result to trickle down from highest rank
        dist_from_highest = (worldsize - 1) - rank

        total_steps = warmup_steps + (num_stages * num_microbatch)


        Rank 0: 0F 0F 0F 0F 2F 2F 2F 2F
        Rank 1:    1F 1F 1F 1F 3F3B 3F 3F 3F
        """

        def minibatch_index(step):
            # Given the step index, find the corresponding minibatch index.

            # equivalent to a triple nested loop like this
            # for sequence_id in range(self.seq_size):
            #     for stage in self.stages:
            #         for microbatch_within_sequence:
            #             ...
            # step: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            # index:0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4,  5,  6,  7,  6,  7
            return (step % self.seq_size) + self.seq_size * int(
                step / (self.seq_size * self.n_local_stages)
            )

        def stage_index(step):
            # step: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            # index:0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,  1,  0,  0,  1,  1
            return int((step / self.seq_size) % self.n_local_stages)

        """

        my theory was that the hang could be fixed if I orchestrate the recvs after the sends from the schedule side, but i should probably
        see if i can prove what caused the hang before i work on it further
        """
        logger.debug(
            f"rank {self.pp_id} - minibatch_index {[minibatch_index(step) for step in range(self.total_steps)]}"
        )
        logger.debug(
            f"rank {self.pp_id} - stage_index {[stage_index(step) for step in range(self.total_steps)]}"
        )

        forward_batched_op_handle: Optional[dist.Work] = None
        backward_batched_op_handle: Optional[dist.Work] = None

        # edge case for first stage on each rank we need to call receive, recv for future microbatches will be fetched after fwd
        # TODO: move this to its own class? `OneTimeUseRecv`?
        forward_first_recv: Optional[List[dist.P2POp]] = self.stages[
            0
        ].get_fwd_recv_ops()

        # edge case for the last stage on each rank we need to call receive, recv for future microbatches will be fetched after bwd
        backward_first_recv: Optional[List[dist.P2POp]] = self.stages[
            -1
        ].get_bwd_recv_ops()

        backward_stages = list(reversed(self.stages))
        for step in range(self.total_steps):
            mb_id_fwd = minibatch_index(step)
            fwd_stage_id = stage_index(step)
            forward_stage = self.stages[fwd_stage_id]
            fwd_stage_id_next = None
            forward_stage_next = None

            backward_step = step - self.warmup_steps
            mb_id_bwd = minibatch_index(backward_step)
            bwd_stage_id = stage_index(backward_step)
            bwd_stage_id_next = None
            backward_stage_next = None
            backward_stage = backward_stages[bwd_stage_id]

            # info for next stages
            if step < self.total_steps:
                fwd_stage_id_next = stage_index(step + 1)
                forward_stage_next = self.stages[fwd_stage_id_next]
                bwd_stage_id_next = stage_index(backward_step + 1)
                backward_stage_next = backward_stages[bwd_stage_id_next]

            if step < self.forward_steps:
                if forward_first_recv:
                    logger.debug(
                        f"rank {self.pp_id} - forward edge case for first stage"
                    )
                    dist.batch_isend_irecv(forward_first_recv).pop().wait()
                    forward_first_recv = None

                if forward_batched_op_handle:
                    logger.debug(
                        f"rank: {self.pp_id} - waiting on batched_op_handle before fwd"
                    )
                    forward_batched_op_handle.wait()
                    forward_batched_op_handle = None

                with record_function(f"Stage {forward_stage.stage_id} Forward"):
                    logger.debug(
                        f"pp_id {self.pp_id} step {step} forward_stage {forward_stage.stage_id} mb_id {mb_id_fwd}"
                    )
                    forward_stage.forward(microbatches[mb_id_fwd])

                requests: List[dist.P2POp] = []

                # send output activations if this is not the last stage
                ops = forward_stage.get_fwd_send_ops()
                requests.extend(ops)

                # add recv for the NEXT stage, do not do this for last stage
                if forward_stage_next is not None:
                    ops = forward_stage_next.get_fwd_recv_ops()
                    if mb_id_fwd != len(microbatches) - 1:
                        requests.extend(ops)

                if requests:
                    logger.debug(
                        f"rank: {self.pp_id}, current stage_id {self.stage_id_to_global_stage_id[fwd_stage_id]}, - {[(req.op, req.peer) for req in requests]}"
                    )
                    forward_batched_op_handle = dist.batch_isend_irecv(
                        requests
                    ).pop()

            if step >= self.warmup_steps:
                if backward_first_recv:
                    logger.debug(
                        f"rank {self.pp_id} - backward edge case for last stage"
                    )
                    dist.batch_isend_irecv(backward_first_recv).pop().wait()
                    backward_first_recv = None

                if backward_batched_op_handle:
                    logger.debug(
                        f"rank: {self.pp_id} - waiting on batched_op_handles before bwd"
                    )
                    backward_batched_op_handle.wait()
                    backward_batched_op_handle = None

                with record_function(
                    f"Stage {backward_stage.stage_id} Backward"
                ):
                    logger.debug(
                        f"pp_id {self.pp_id} step {step}/{self.total_steps} backward_step {backward_step} backward_stage_id {backward_stage.stage_id} mb_id {mb_id_bwd}"
                    )
                    backward_stage.backward()

                requests = []

                # send bwd grad if this is not the first stage
                ops = backward_stage.get_bwd_send_ops()
                requests.extend(ops)

                # add recv for the NEXT stage, do not do this for first stage
                if backward_stage_next is not None:
                    ops = backward_stage_next.get_bwd_recv_ops()
                    if mb_id_bwd != len(microbatches) - 1:
                        requests.extend(ops)

                if requests:
                    logger.debug(
                        f"rank: {self.pp_id} - {[(req.op, req.peer) for req in requests]}"
                    )
                    backward_batched_op_handle = dist.batch_isend_irecv(
                        requests
                    ).pop()

        logger.debug("Step exiting")
