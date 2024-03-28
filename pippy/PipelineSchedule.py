# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.profiler import record_function

from pippy.PipelineStage import PipelineStageBase

logger = logging.getLogger(__name__)


class PipelineSchedule(ABC):
    def __init__(
        self,
        stage: PipelineStageBase,
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
    ):
        self._stage = stage
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn
        self._has_backward = self._loss_fn is not None
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward
        self._should_compute_loss: bool = (
            self._stage.is_last and self._loss_fn is not None
        )
        logger.debug(
            f"[{self._stage.stage_index}] Should compute loss: {self._should_compute_loss}"
        )

    @abstractmethod
    def step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).

        kwargs: keyword arguments to the model (as in non-pipeline case).
        """
        raise NotImplementedError


def sorted_batch_isend_irecv(p2p_ops: List[dist.P2POp]) -> Dict[int, dist.Work]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: Dict[int, List[dist.P2POp]] = defaultdict(list)
    work_by_peer: Dict[int, dist.Work] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = dist.batch_isend_irecv(ops).pop()

    return work_by_peer


class PipelineScheduleGPipe(PipelineSchedule):
    def step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        # Pre-process inputs
        if arg_mbs is not None:
            assert len(arg_mbs) == self._n_microbatches
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            assert len(kwarg_mbs) == self._n_microbatches
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if self._should_compute_loss:
            if target_mbs is None:
                raise RuntimeError(
                    "target_mbs must be passed in if loss_fn is not None"
                )
            if len(target_mbs) != self._n_microbatches:
                raise RuntimeError(
                    f"target_mbs length {len(target_mbs)} does not match number of microbatches {self._n_microbatches}"
                )

        # Internal loss container
        internal_losses = []
        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                works = sorted_batch_isend_irecv(ops)
                for work in works.values():
                    work.wait()

                output = self._stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])

                ops = self._stage.get_fwd_send_ops()
                works = sorted_batch_isend_irecv(ops)
                fwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Forwarded microbatch {i}"
            )

            if self._should_compute_loss:
                target = target_mbs[i]  # type: ignore[index]
                if target.shape != output.shape:
                    raise RuntimeError(
                        f"target_mbs[{i}] shape {target.shape} does not match output shape {output.shape}"
                    )
                loss = self._loss_fn(output, target)  # type: ignore[misc]
                internal_losses.append(loss)
                logger.debug(
                    f"[{self._stage.stage_index}] Loss of microbatch {i}: {loss}"
                )

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            work.wait()

        # No loss function, no need to run backward
        if not self._has_backward:
            return

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: List[dist.Work] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops()
                works = sorted_batch_isend_irecv(ops)
                for work in works.values():
                    work.wait()

                loss = internal_losses[i] if len(internal_losses) > 0 else None
                self._stage.backward_one_chunk(loss=loss)

                ops = self._stage.get_bwd_send_ops()
                works = sorted_batch_isend_irecv(ops)
                bwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Backwarded microbatch {i}"
            )

        # Return losses if there is a container passed in
        if losses is not None:
            assert isinstance(
                losses, list
            ), f"losses must be a list but got a {type(losses)}"
            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(internal_losses)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._stage.split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = torch.tensor_split(target, self._n_microbatches)
        else:
            targets_split = None

        # Run microbatches
        self.step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        return self._stage.merge_outputs()


class PipelineSchedule1F1B(PipelineSchedule):
    def __init__(self, stage: PipelineStageBase):
        self._stage = stage
        self.stage_index = stage.stage_index
        self.rank = stage.rank
        self.pp_group_size = stage.world_size

    def step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
    ):
        if arg_mbs is not None:
            # TODO: fix this so it is preset
            self._n_microbatches = len(arg_mbs)
            assert len(arg_mbs) == self._n_microbatches
        else:
            arg_mbs = [()] * self._n_microbatches

        # forward for num_microbatches + backward for num_microbatches
        total_ops = self._n_microbatches * 2

        # Example, 4 GPUs, 8 microbatches
        # Stage 0: 6 warmup, 2 1f1b, 6 cooldown
        # Stage 1: 4 warmup, 4 1f1b, 4 cooldown
        # Stage 2: 2 warmup, 6 1f1b, 2 cooldown
        # Stage 3: 0 warmup, 8 1f1b, 0 cooldown
        # fwd only
        warmup_steps = min(
            self._n_microbatches,
            2 * (self.pp_group_size - self.stage_index - 1),
        )

        # fwd + bwd
        main_1f1b_steps = self._n_microbatches - warmup_steps

        # bwd only
        cooldown_steps = total_ops - (warmup_steps + (2 * main_1f1b_steps))

        total_steps = warmup_steps + main_1f1b_steps + cooldown_steps

        logger.debug(
            f"""
            Rank {self.rank}:
            Warmup steps: {warmup_steps}
            Main 1F1B steps: {main_1f1b_steps}
            Cooldown steps: {cooldown_steps}
            Total steps: {total_steps}
        """
        )

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []
        bwd_sends_to_wait: List[dist.Work] = []

        for i in range(total_steps):
            if i < self._n_microbatches:
                # forward
                with record_function(f"Forward {i}"):
                    ops = self._stage.get_fwd_recv_ops()
                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    self._stage.forward_one_chunk(arg_mbs[i])

                    ops = self._stage.get_fwd_send_ops()
                    works = sorted_batch_isend_irecv(ops)
                    fwd_sends_to_wait.extend(works.values())

            if (
                warmup_steps
                <= i
                < warmup_steps + main_1f1b_steps + cooldown_steps
            ):
                # backward
                with record_function(f"Backward {i}"):
                    ops = self._stage.get_bwd_recv_ops()
                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    self._stage.backward_one_chunk()

                    ops = self._stage.get_bwd_send_ops()
                    works = sorted_batch_isend_irecv(ops)
                    bwd_sends_to_wait.extend(works.values())

        # Wait for all forward sends to finish
        for work in fwd_sends_to_wait:
            work.wait()

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()

    def step(self, *args, **kwargs):
        # TODO
        pass


class PipelineScheduleLoopedBFS(PipelineSchedule):
    def __init__(self, stages: List[PipelineStageBase]):
        self._stages = stages

    def step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
    ):
        # Pre-process inputs
        if arg_mbs is not None:
            # TODO: fix this so it is preset
            self._n_microbatches = len(arg_mbs)
            assert len(arg_mbs) == self._n_microbatches
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            assert len(kwarg_mbs) == self._n_microbatches
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        for s, stage in enumerate(self._stages):
            for i in range(self._n_microbatches):
                with record_function(f"Stage {s} Forward"):
                    ops = stage.get_fwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])

                    ops = stage.get_fwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)

        for stage in reversed(self._stages):
            for i in range(self._n_microbatches):
                with record_function(f"Stage {stage.stage_index} Backward"):
                    ops = stage.get_bwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    stage.backward_one_chunk(chunk=i)

                    ops = stage.get_bwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)

    def step(self, *args, **kwargs):
        # TODO
        pass


class PipelineScheduleInterleaved1F1B(PipelineSchedule):
    def __init__(self, stages: List[PipelineStageBase]):
        if len(stages) <= 1:
            raise ValueError(
                "Looped DFS schedule requires at least two stages to be used."
            )

        self.stages = stages
        self.n_local_stages = len(stages)
        stage = stages[0]
        self.pp_group_size = stage.world_size
        self.rank = stage.rank
        self.total_stages = self.n_local_stages * self.pp_group_size
        self.local_idx_to_global_stage_id = [
            stage.stage_index for stage in self.stages
        ]

    def step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
    ):
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
        if arg_mbs is not None:
            # TODO: fix this so it is preset
            self._n_microbatches = len(arg_mbs)
            assert len(arg_mbs) == self._n_microbatches
        else:
            arg_mbs = [()] * self._n_microbatches

        if self._n_microbatches % self.pp_group_size != 0:
            raise ValueError(
                f"Looped DFS schedule requires the number of microbatches ({self._n_microbatches}) \
                to be a multiple of the number of pipelined ranks ({self.pp_group_size})."
            )

        # warmup steps for latest pp stage is trivial to compute
        # increment warmup_steps by 2 for each hop away
        warmup_steps = (self.n_local_stages - 1) * self.pp_group_size
        warmup_steps += 2 * ((self.pp_group_size - 1) - self.rank)
        fwd_bwd_steps = (
            self.n_local_stages * self._n_microbatches
        ) - warmup_steps
        cooldown_steps = (
            self.n_local_stages * self._n_microbatches
        ) - fwd_bwd_steps

        assert (
            warmup_steps + fwd_bwd_steps * 2 + cooldown_steps
            == self.n_local_stages * self._n_microbatches * 2
        )
        self.total_steps = warmup_steps + fwd_bwd_steps + cooldown_steps

        logger.debug(
            f"""
            rank {self.rank}
            warmup_steps {warmup_steps}
            1f1b {fwd_bwd_steps}
            cooldown_steps {cooldown_steps}
            """
        )

        def microbatch_index(step):
            # Given the step index, find the corresponding microbatch index.

            # equivalent to a triple nested loop like this          ...
            # for gpu in range(self.pp_group_size):
            #     for stage in self.stages:
            #         for microbatch_within_sequence:
            #             ...
            return (step % self.pp_group_size) + self.pp_group_size * int(
                step / (self.pp_group_size * self.n_local_stages)
            )

        def forward_stage_local_index(step):
            return (step // self.pp_group_size) % self.n_local_stages

        def backward_stage_local_index(step):
            return (
                self.n_local_stages
                - 1
                - ((step - warmup_steps) // self.pp_group_size)
                % self.n_local_stages
            )

        for step in range(self.total_steps):
            # warmup, forward only
            if step < warmup_steps:
                fwd_stage = self.stages[forward_stage_local_index(step)]
                mb_index = microbatch_index(step)
                logger.debug(
                    f"{self.rank}: {step=}, {fwd_stage.stage_index=}, {mb_index=}"
                )

                with record_function(f"Forward {step}"):
                    ops = fwd_stage.get_fwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    fwd_stage.forward(arg_mbs[mb_index])

                    ops = fwd_stage.get_fwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)
            # 1f1b
            elif warmup_steps <= step < warmup_steps + fwd_bwd_steps:
                fwd_stage = self.stages[forward_stage_local_index(step)]
                bwd_stage = self.stages[backward_stage_local_index(step)]
                logger.debug(
                    f"{self.rank}: {step=}, {fwd_stage.stage_index=}, {bwd_stage.stage_index=}, {mb_index=}"
                )
                with record_function(f"1F1B {step}"):
                    ops = fwd_stage.get_fwd_recv_ops()
                    ops.extend(bwd_stage.get_bwd_recv_ops())

                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    fwd_stage.forward_one_chunk(arg_mbs[mb_index])
                    bwd_stage.backward_one_chunk()

                    ops = fwd_stage.get_fwd_send_ops()
                    ops.extend(bwd_stage.get_bwd_send_ops())
                    if ops:
                        dist.batch_isend_irecv(ops)
            # cooldown
            else:
                bwd_stage = self.stages[backward_stage_local_index(step)]
                logger.debug(
                    f"{self.rank}: {step=}, {bwd_stage.stage_index=}, {mb_index=}"
                )
                with record_function(f"Cooldown (backward) {step}"):
                    ops = bwd_stage.get_bwd_recv_ops()

                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    bwd_stage.backward_one_chunk()

                    ops = bwd_stage.get_bwd_send_ops()

                    if ops:
                        dist.batch_isend_irecv(ops)

    def step(self, *args, **kwargs):
        # TODO
        pass
