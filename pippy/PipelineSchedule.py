# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.profiler import record_function

from ._IR import Pipe
from ._PipelineStage import PipelineStageBase
from .microbatch import merge_chunks, split_args_kwargs_into_chunks

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class PipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn
        self._output_merge_spec = output_merge_spec
        # Derived
        self._has_backward = self._loss_fn is not None
        # To be filled by subclasses
        self._pipe_info: Optional[Pipe.PipeInfo] = None

        # Holds the losses for each microbatch.
        self._internal_losses: List[torch.Tensor] = []
        logger.info(f"Using {self.__class__.__name__}")

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        if stage.is_last and self._has_backward:
            loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
            self._internal_losses.append(loss)
            logger.debug(
                f"[{stage.stage_index}] Loss of microbatch {mb_index}: {loss}"
            )

    def _maybe_get_loss(self, stage, mb_index):
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._has_backward and valid_index:
            return self._internal_losses[mb_index]
        elif len(self._internal_losses) != 0 and not valid_index:
            raise RuntimeError(
                f"Loss for microbatch {mb_index} is not available. "
                f"Available losses for microbatches: {self._internal_losses}"
            )
        else:
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # if stages not a list turn into a list
        if not isinstance(stages, list):
            stages = [stages]
        contains_last_stage = any([stage.is_last for stage in stages])

        # Return losses if there is a container passed in
        if contains_last_stage and losses is not None:
            if len(self._internal_losses) != self._n_microbatches:
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}"
                )

            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(self._internal_losses)

        self._internal_losses.clear()

    @abstractmethod
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
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
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        raise NotImplementedError

    def _check_inputs(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            if len(mbs) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(mbs)}"
                )

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            if not isinstance(losses, list):
                raise TypeError(
                    f"losses must be a list but got a {type(losses)}"
                )

        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output, target):
        return self._loss_fn(output, target)  # type: ignore[misc]

    def _split_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if self._pipe_info is not None:
            # Use spec from `pipe_info`
            args_chunk_spec = self._pipe_info.args_chunk_spec
            kwargs_chunk_spec = self._pipe_info.kwargs_chunk_spec
        else:
            # Use default spec from `microbatch.py` (i.e. chunk dim 0 for each arg/kwarg)
            args_chunk_spec = None
            kwargs_chunk_spec = None

        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                args_chunk_spec,
                kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _merge_outputs(self, output_chunks: List[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


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


class PipelineScheduleSingle(PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stage: PipelineStageBase,
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )
        self._pipe_info = (
            stage.pipe_info if hasattr(stage, "pipe_info") else None  # type: ignore[attr-defined]
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(
                torch.tensor_split(target, self._n_microbatches)
            )
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        if self._stage.is_last:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None


class ScheduleGPipe(PipelineScheduleSingle):
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                works = sorted_batch_isend_irecv(ops)
                for work in works.values():
                    work.wait()

                output = self._stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops()
                works = sorted_batch_isend_irecv(ops)
                fwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Forwarded microbatch {i}"
            )

            self._maybe_compute_loss(self._stage, output, target_mbs, i)

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
            # set library-specific data-parallel config flags to ensure gradient accumulation across microbatches
            self._stage._configure_data_parallel_mode(
                i == self._n_microbatches - 1
            )

            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops()
                works = sorted_batch_isend_irecv(ops)
                for work in works.values():
                    work.wait()

                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(loss=loss)

                ops = self._stage.get_bwd_send_ops()
                works = sorted_batch_isend_irecv(ops)
                bwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Backwarded microbatch {i}"
            )

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()


class Schedule1F1B(PipelineScheduleSingle):
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )

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
            2 * (self._num_stages - self._stage.stage_index - 1),
        )
        # fwd + bwd
        main_1f1b_steps = self._n_microbatches - warmup_steps
        # bwd only
        cooldown_steps = total_ops - (warmup_steps + (2 * main_1f1b_steps))
        total_steps = warmup_steps + main_1f1b_steps + cooldown_steps
        logger.debug(
            f"Stage {self._stage.stage_index}: "
            f"Warmup steps: {warmup_steps}, "
            f"Main 1F1B steps: {main_1f1b_steps}, "
            f"Cooldown steps: {cooldown_steps}, "
            f"Total steps: {total_steps}"
        )

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []
        bwd_sends_to_wait: List[dist.Work] = []

        # bwd chunk counter
        bwd_mb_index = 0
        self._stage._configure_data_parallel_mode(last_backward=False)
        for i in range(total_steps):
            if i < self._n_microbatches:
                # forward
                with record_function(f"Forward {i}"):
                    ops = self._stage.get_fwd_recv_ops()
                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    output = self._stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                    ops = self._stage.get_fwd_send_ops()
                    works = sorted_batch_isend_irecv(ops)
                    fwd_sends_to_wait.extend(works.values())

                self._maybe_compute_loss(self._stage, output, target_mbs, i)

            if i >= warmup_steps and self._has_backward:
                self._stage._configure_data_parallel_mode(
                    last_backward=(i == total_steps - 1)
                )

                # backward
                with record_function(f"Backward {bwd_mb_index}"):
                    ops = self._stage.get_bwd_recv_ops()
                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    loss = self._maybe_get_loss(self._stage, bwd_mb_index)
                    self._stage.backward_one_chunk(loss=loss)

                    ops = self._stage.get_bwd_send_ops()
                    works = sorted_batch_isend_irecv(ops)
                    bwd_sends_to_wait.extend(works.values())
                    bwd_mb_index += 1

        # Wait for all forward sends to finish
        for work in fwd_sends_to_wait:
            work.wait()

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)


class PipelineScheduleMulti(PipelineSchedule):
    """
    Base class for multi-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stages: List[PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        if len(stages) <= 1:
            raise ValueError(
                f"Multi-stage schedule expects at least two stages but got {len(stages)}"
            )
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )
        self._pipe_info = (
            stages[0].pipe_info if hasattr(stages[0], "pipe_info") else None  # type: ignore[attr-defined]
        )
        # Self attributes
        self._stages = stages
        self._num_stages = stages[0].num_stages
        # Set the same has_backward flag for stage object
        for stage in self._stages:
            stage.has_backward = self._has_backward

        self._should_compute_loss = (
            lambda stage: stage.is_last and self._loss_fn is not None
        )

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        # Clean per iteration
        for stage in self._stages:
            stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(
                torch.tensor_split(target, self._n_microbatches)
            )
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        for stage in self._stages:
            if stage.is_last:
                return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage
        return None


class ScheduleLoopedBFS(PipelineScheduleMulti):
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,  # TODO
        losses: Optional[List] = None,  # TODO
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

        for stage in self._stages:
            for i in range(self._n_microbatches):
                with record_function(f"Stage {stage.stage_index} Forward"):
                    ops = stage.get_fwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    output = stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])
                    self._maybe_compute_loss(stage, output, target_mbs, i)

                    ops = stage.get_fwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)

        for stage in reversed(self._stages):
            for i in range(self._n_microbatches):
                stage._configure_data_parallel_mode(
                    i == self._n_microbatches - 1
                )
                with record_function(f"Stage {stage.stage_index} Backward"):
                    ops = stage.get_bwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    loss = self._maybe_get_loss(stage, i)
                    stage.backward_one_chunk(loss=loss)

                    ops = stage.get_bwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)

        self._update_losses(self._stages, losses)


class ScheduleInterleaved1F1B(PipelineScheduleMulti):
    def __init__(
        self,
        stages: List[PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        self.pp_group_size = stages[0].group_size
        # TODO: is this limitation a must?
        if n_microbatches % self.pp_group_size != 0:
            raise ValueError(
                "Interleaved 1F1B requires the number of microbatches to be a "
                f"multiple of the number of pipeline ranks ({self.pp_group_size}), "
                f"but got {n_microbatches}."
            )

        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )

        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Operate on the microbatches for interleaved 1f1b schedule (https://arxiv.org/pdf/2104.04473.pdf).

        Highest rank has a warmup (fwd only) count of [len(stages) - 1] * number of PP ranks
        and each rank away from highest rank adds 2 warmup steps due to:
            - one happened before highest rank's warmup started,
            - one waiting for backward result to trickle down from highest rank

        TODO: Interleaved 1F1B does not support using sorted_batch_isend_irecv()
        because it requires recvs and sends from different peers
        to execute in the same coalesced operation. As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )

        # increment warmup_steps by 2 for each hop away
        warmup_steps = (self.n_local_stages - 1) * self.pp_group_size
        warmup_steps += 2 * ((self.pp_group_size - 1) - self.rank)
        warmup_steps = min(
            warmup_steps, self._n_microbatches * self.n_local_stages
        )
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
        total_steps = warmup_steps + fwd_bwd_steps + cooldown_steps

        logger.debug(
            f"""
            rank {self.rank}
            warmup_steps {warmup_steps}
            1f1b {fwd_bwd_steps}
            cooldown_steps {cooldown_steps}
            """
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

        fwd_stage_mb_index: Dict[PipelineStageBase, int] = defaultdict(int)
        bwd_stage_mb_index: Dict[PipelineStageBase, int] = defaultdict(int)

        # Delay send waits
        sends_to_wait: List[dist.Work] = []

        # Store ops (potentially across steps)
        ops: List[dist.P2POp] = []

        # Warmup Phase (forward only)
        for step in range(warmup_steps):
            fwd_stage = self._stages[forward_stage_local_index(step)]

            # This will assign the current microbatch index and update it for future steps
            fwd_stage_mb_index[fwd_stage] = (
                mb_index := fwd_stage_mb_index[fwd_stage]
            ) + 1

            logger.debug(
                f"Rank {self.rank}: {step=}, {fwd_stage.stage_index=}, {mb_index=}"
            )

            with record_function(f"Forward {step}"):
                ops.extend(fwd_stage.get_fwd_recv_ops())
                if ops:
                    work = dist.batch_isend_irecv(ops).pop()
                    work.wait()
                    ops.clear()

                output = fwd_stage.forward_one_chunk(arg_mbs[mb_index], kwarg_mbs[mb_index])  # type: ignore[index]

                ops.extend(fwd_stage.get_fwd_send_ops())
                # If we are right before the fwd-bwd step, then we need to delay the send to the next step,
                # This is because fwd-bwd send/recvs among ranks need to be aligned to prevent a hang.
                # In the edge cases where there are no fwd_bwds and cooldown is immediate, then no delay is needed
                if ops and (step != warmup_steps - 1 or fwd_bwd_steps == 0):
                    work = dist.batch_isend_irecv(ops).pop()
                    sends_to_wait.append(work)
                    ops.clear()

                self._maybe_compute_loss(
                    fwd_stage, output, target_mbs, mb_index
                )

        # 1F1B Phase (forward and backward)
        for step in range(warmup_steps, warmup_steps + fwd_bwd_steps):
            fwd_stage = self._stages[forward_stage_local_index(step)]
            bwd_stage = self._stages[backward_stage_local_index(step)]

            fwd_stage_mb_index[fwd_stage] = (
                fwd_mb_index := fwd_stage_mb_index[fwd_stage]
            ) + 1
            bwd_stage_mb_index[bwd_stage] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage]
            ) + 1

            bwd_stage._configure_data_parallel_mode(
                bwd_mb_index == self._n_microbatches - 1
            )
            logger.debug(
                f"Rank {self.rank}: {step=}, {fwd_stage.stage_index=}, {bwd_stage.stage_index=}, {fwd_mb_index=}, {bwd_mb_index=}"
            )
            with record_function(f"1F1B {step}"):
                ops.extend(fwd_stage.get_fwd_recv_ops())
                ops.extend(bwd_stage.get_bwd_recv_ops())
                if ops:
                    work = dist.batch_isend_irecv(ops).pop()
                    work.wait()
                    ops.clear()

                # Forward
                output = fwd_stage.forward_one_chunk(arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]
                ops.extend(fwd_stage.get_fwd_send_ops())
                self._maybe_compute_loss(
                    fwd_stage, output, target_mbs, fwd_mb_index
                )

                # Backward
                loss = self._maybe_get_loss(bwd_stage, bwd_mb_index)
                bwd_stage.backward_one_chunk(loss=loss)
                ops.extend(bwd_stage.get_bwd_send_ops())

        # Cooldown Phase (backward only)
        for step in range(warmup_steps + fwd_bwd_steps, total_steps):
            bwd_stage = self._stages[backward_stage_local_index(step)]
            bwd_stage_mb_index[bwd_stage] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage]
            ) + 1
            bwd_stage._configure_data_parallel_mode(
                bwd_mb_index == self._n_microbatches - 1
            )

            logger.debug(
                f"Rank {self.rank}: {step=}, {bwd_stage.stage_index=}, {bwd_mb_index=}"
            )
            with record_function(f"Cooldown {step}"):
                ops.extend(bwd_stage.get_bwd_recv_ops())
                if ops:
                    work = dist.batch_isend_irecv(ops).pop()
                    work.wait()
                    ops.clear()

                loss = self._maybe_get_loss(bwd_stage, bwd_mb_index)
                bwd_stage.backward_one_chunk(loss=loss)

                ops.extend(bwd_stage.get_bwd_send_ops())
                if ops:
                    work = dist.batch_isend_irecv(ops).pop()
                    sends_to_wait.append(work)
                    ops.clear()

        # Make sure all sends are finished
        for work in sends_to_wait:
            work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)


class ScheduleDoraPP(PipelineScheduleMulti):
    """
    This is interleaved dfs+bfs zero bubble schedule.
    """
    def __init__(
        self,
        stages: List[PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        self.pp_group_size = stages[0].group_size
        # TODO: is this limitation a must?
        if n_microbatches % self.pp_group_size != 0:
            raise ValueError(
                "Interleaved 1F1B requires the number of microbatches to be a "
                f"multiple of the number of pipeline ranks ({self.pp_group_size}), "
                f"but got {n_microbatches}."
            )

        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )

        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
        microbatch_size: Optional[int] = None,
        model_dim: Optional[int] = None,
    ):
        """
        Operate on the microbatches for doraPP schedule .

        Highest rank has a warmup (fwd only) count of [len(stages) - 1] * number of PP ranks
        and each rank away from highest rank adds 2 warmup steps due to:
            - one happened before highest rank's warmup started,
            - one waiting for backward result to trickle down from highest rank

        TODO: Interleaved 1F1B does not support using sorted_batch_isend_irecv()
        because it requires recvs and sends from different peers
        to execute in the same coalesced operation. As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )

        # increment warmup_steps by 2 for each hop away
        warmup_steps = (self.n_local_stages - 1) * self.pp_group_size
        warmup_steps += 2 * ((self.pp_group_size - 1) - self.rank)
        warmup_steps = min(
            warmup_steps, self._n_microbatches * self.n_local_stages
        )
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
        total_steps = warmup_steps + fwd_bwd_steps + cooldown_steps

        logger.debug(
            f"""
            n_microbatches {self._n_microbatches}
            stages {self.n_local_stages}
            rank {self.rank}
            warmup_steps {warmup_steps}
            1f1b {fwd_bwd_steps}
            cooldown_steps {cooldown_steps}
            """
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

        fwd_stage_mb_index: Dict[PipelineStageBase, int] = defaultdict(int)
        bwd_stage_mb_index: Dict[PipelineStageBase, int] = defaultdict(int)

        # Delay send waits
        sends_to_wait: List[dist.Work] = []

        # Store ops (potentially across steps)
        ops: List[dist.P2POp] = []

        # Warmup Phase (forward only)
        for step in range(warmup_steps):
            fwd_stage = self._stages[forward_stage_local_index(step)]

            # This will assign the current microbatch index and update it for future steps
            fwd_stage_mb_index[fwd_stage] = (
                mb_index := fwd_stage_mb_index[fwd_stage]
            ) + 1

            with record_function(f"Forward {step}"):
                ops.extend(fwd_stage.get_fwd_recv_ops())
                if ops:
                    work = dist.batch_isend_irecv(ops).pop()
                    work.wait()
                    ops.clear()

                output = fwd_stage.forward_one_chunk(arg_mbs[mb_index], kwarg_mbs[mb_index])  # type: ignore[index]

                ops.extend(fwd_stage.get_fwd_send_ops())
                # If we are right before the fwd-bwd step, then we need to delay the send to the next step,
                # This is because fwd-bwd send/recvs among ranks need to be aligned to prevent a hang.
                # In the edge cases where there are no fwd_bwds and cooldown is immediate, then no delay is needed
                if ops and (step != warmup_steps - 1 or fwd_bwd_steps == 0):
                    work = dist.batch_isend_irecv(ops).pop()
                    sends_to_wait.append(work)
                    ops.clear()

                self._maybe_compute_loss(
                    fwd_stage, output, target_mbs, mb_index
                )

        # 1F1B Phase (forward and backward)
        for step in range(warmup_steps, warmup_steps + fwd_bwd_steps):
            fwd_stage = self._stages[forward_stage_local_index(step)]
            bwd_stage = self._stages[backward_stage_local_index(step)]

            fwd_stage_mb_index[fwd_stage] = (
                fwd_mb_index := fwd_stage_mb_index[fwd_stage]
            ) + 1
            bwd_stage_mb_index[bwd_stage] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage]
            ) + 1

            bwd_stage._configure_data_parallel_mode(
                bwd_mb_index == self._n_microbatches - 1
            )
            logger.debug(
                f"Rank {self.rank}: {step=}, {fwd_stage.stage_index=}, {bwd_stage.stage_index=}, {fwd_mb_index=}, {bwd_mb_index=}"
            )
            with record_function(f"1F1B {step}"):
                ops.extend(fwd_stage.get_fwd_recv_ops())
                ops.extend(bwd_stage.get_bwd_recv_ops())
                if ops:
                    work = dist.batch_isend_irecv(ops).pop()
                    work.wait()
                    ops.clear()

                # Forward
                output = fwd_stage.forward_one_chunk(arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]
                ops.extend(fwd_stage.get_fwd_send_ops())
                self._maybe_compute_loss(
                    fwd_stage, output, target_mbs, fwd_mb_index
                )

                # Backward
                loss = self._maybe_get_loss(bwd_stage, bwd_mb_index)
                bwd_stage.backward_one_chunk(loss=loss)
                ops.extend(bwd_stage.get_bwd_send_ops())

        # Cooldown Phase (backward only)
        for step in range(warmup_steps + fwd_bwd_steps, total_steps):
            bwd_stage = self._stages[backward_stage_local_index(step)]
            bwd_stage_mb_index[bwd_stage] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage]
            ) + 1
            bwd_stage._configure_data_parallel_mode(
                bwd_mb_index == self._n_microbatches - 1
            )

            logger.debug(
                f"Rank {self.rank}: {step=}, {bwd_stage.stage_index=}, {bwd_mb_index=}"
            )
            with record_function(f"Cooldown {step}"):
                ops.extend(bwd_stage.get_bwd_recv_ops())
                if ops:
                    work = dist.batch_isend_irecv(ops).pop()
                    work.wait()
                    ops.clear()

                loss = self._maybe_get_loss(bwd_stage, bwd_mb_index)
                bwd_stage.backward_one_chunk(loss=loss)

                ops.extend(bwd_stage.get_bwd_send_ops())
                if ops:
                    work = dist.batch_isend_irecv(ops).pop()
                    sends_to_wait.append(work)
                    ops.clear()

        # Make sure all sends are finished
        for work in sends_to_wait:
            work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)









        ##########################################################
        #  Dora PP Xlformer Implementation
        ##########################################################

        input_tensors = [[] for _ in range(len(self._stages))]
        output_tensors = [[] for _ in range(len(self._stages))]
        output_tensor_grads = [[] for _ in range(len(self._stages))]
        # We need to pop input, output and grad during bwd, we use this list to track real input tensor index.
        popped_input_tensors = [[] for _ in range(len(self._stages))]
        input_tensor_grad = None

        pipeline_parallel_size = self.pp_group_size
        pipeline_parallel_rank = self._stage.stage_index

        microbatch_x = arg_mbs
        microbatch_y = target_mbs
        microbatch_mask = None
        mask = None
        if mask is not None:
            microbatch_mask = mask.split(args.pipeline_parallel_microbatch_size, dim=0)

        num_microbatches = self._n_microbatches
        # microbatch_attn_bias = [
        #     model[0].get_attn_bias(microbatch_x[i], cache=None)
        #     for i in range(num_microbatches)
        # ]
        microbatch_attn_bias = [
            self._stages[0].submodule.get_attn_bias(microbatch_x[i], cache=None)
            for i in range(num_microbatches)
        ]


        # TODO: get the model args from API directly, should modify it later
        assert(microbatch_size is not None), "microbatch_size is None"
        assert(model_dim is not None), "model_dim is None"

        microbatch_less_than_pp = num_microbatches < pipeline_parallel_size
        num_round = max(num_microbatches // pipeline_parallel_size, 1)
        assert (
            num_microbatches % num_round == 0
        ), "Number of microbatches should be divisible by number of pipeline rounds."
        # the number of microbatches run in each round, in dfs it is pipeline_parallel_size
        num_microbatch_per_round = num_microbatches // num_round

        tensor_shape = (
            microbatch_size,
            model_dim,
        )

        num_model_chunks = len(model)
        total_num_microbatches = num_microbatches * num_model_chunks

        dtype = get_torch_dtype(args.dtype)

        mpu.set_virtual_pipeline_model_parallel_rank(0)
        all_warmup_microbatches = False

        if not args.model.enable_ddp:
            for model_chunk in model:
                model_chunk._rebuild_full_params_recursive()
        else:
            for model_chunk in model:
                model_chunk.zero_grad()


        num_warmup_microbatches = 0
        # The number of microbatches that last pipeline stage run before 1f1b.
        num_warmup_microbatches += (num_model_chunks - 1) * num_microbatch_per_round
        # From last PP stage up, each rank will be 2 more than the previous one.
        num_warmup_microbatches += (
            pipeline_parallel_size - pipeline_parallel_rank - 1
        ) * 2
        num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
        num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
        # The number of 1f1b for zero bubble schedule
        if num_microbatches == pipeline_parallel_size:
            num_1f1b_microbatches = pipeline_parallel_rank
        else:
            num_1f1b_microbatches = 2 * pipeline_parallel_rank

        # Checkpoint the activations of partial Transformer layers in a number of micro-batches
        # within the maximum outstanding micro-batch backpropagations.
        # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
        # checkpoint partial Transformer layers (or skip checkpointing) and
        # the rest of micro-batches within a window of micro-batches checkpoint
        # all Transformer layers. The window of micro-batches is set by the maximum
        # outstanding backpropagations and becomes smaller at later pipeline stages.
        # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
        max_outstanding_backprops = None
        if args.num_microbatches_with_partial_activation_checkpoints is not None:
            max_outstanding_backprops = num_warmup_microbatches + 1

        p0_chunk0_batch = [0, 0]
        mean_losses = []

        def get_model_chunk_id(microbatch_id, forward):
            """Helper method to get the model chunk ID given the iteration number.
            Each group has num_microbatch_per_round * num_model_chunks microbatches.
            within each chunk, there are num_microbatch_per_round microbatches.
            backward is reverse order of forward.
            """
            microbatch_id_in_group = microbatch_id % (
                num_microbatch_per_round * num_model_chunks
            )
            model_chunk_id = microbatch_id_in_group // num_microbatch_per_round
            if not forward:
                model_chunk_id = num_model_chunks - model_chunk_id - 1
            return model_chunk_id

        def get_real_microbatch_id(microbatch_id: int) -> int:
            """Get the microbatch id for input tokens."""
            microbatch_group_size = num_microbatch_per_round * num_model_chunks
            microbatch_group_id = microbatch_id // microbatch_group_size
            real_microbatch_id_in_group = (
                microbatch_id % microbatch_group_size
            ) % num_microbatch_per_round
            real_microbatch_id = (
                real_microbatch_id_in_group + microbatch_group_id * num_microbatch_per_round
            )
            return real_microbatch_id

        def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
            """Check if an iteration is the first for a model chunk."""
            microbatch_group_size = num_microbatch_per_round * num_model_chunks
            microbatch_group_id = microbatch_id // microbatch_group_size
            microbatch_id_in_group = microbatch_id % microbatch_group_size
            if microbatch_group_id == 0:
                return microbatch_id_in_group % num_microbatch_per_round == 0
            else:
                return False

        def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
            """Check if an iteration is the last for a model chunk."""
            microbatch_group_size = num_microbatch_per_round * num_model_chunks
            num_microbatch_groups = total_num_microbatches // microbatch_group_size
            microbatch_group_id = microbatch_id // microbatch_group_size
            microbatch_id_in_group = microbatch_id % microbatch_group_size
            if microbatch_group_id == num_microbatch_groups - 1:
                return (
                    microbatch_id_in_group % num_microbatch_per_round
                    == num_microbatch_per_round - 1
                )
            else:
                return False

        def get_input_index(microbatch_id):
            """Get pipeline input index for a microbatch"""
            microbatch_group_size = num_microbatch_per_round * num_model_chunks
            microbatch_id_in_group = microbatch_id % microbatch_group_size
            microbatch_group_id = microbatch_id // microbatch_group_size
            input_index = microbatch_id_in_group % num_microbatch_per_round
            return input_index + microbatch_group_id * num_microbatch_per_round

        def microbatch_fwd(
            model_chunk_id,
            input_tensor,
            microbatch_tokens,
            y,
            state,
            mask,
            mean_losses,
            is_first_microbatch=False,
            recompute_attn=None,
            recompute_fc1_fc3=None,
            attn_bias=None,
        ):
            if input_tensor is None:
                assert mpu.is_pipeline_first_stage()
            else:
                assert not mpu.is_pipeline_first_stage()

            if args.num_microbatches_with_partial_activation_checkpoints is not None:
                output, _ = model[model_chunk_id](
                    microbatch_tokens,
                    pipeline_parallel_input_tensor=input_tensor,
                    is_first_microbatch=is_first_microbatch,
                    recompute_attn=recompute_attn,
                    recompute_fc1_fc3=recompute_fc1_fc3,
                    precomputed_attn_bias=attn_bias,
                )
            else:
                output, _ = model[model_chunk_id](
                    microbatch_tokens,
                    pipeline_parallel_input_tensor=input_tensor,
                    is_first_microbatch=is_first_microbatch,
                    precomputed_attn_bias=attn_bias,
                )

            if mpu.is_pipeline_last_stage():
                if loss_fn is not None:
                    loss = loss_fn(
                        output,
                        y,
                        mask,
                    )
                    output = loss.mean() / num_microbatches
                else:
                    if args.model.loss_parallel:
                        tok_loss = state.scale * vocab_parallel_cross_entropy(
                            partial_logits=output,
                            target=y,
                            z_loss_multiplier=args.z_loss_multiplier,
                        )
                    else:
                        tok_loss = state.scale * F.cross_entropy(
                            output.flatten(0, 1), y.flatten(0, 1), reduction="none"
                        )
                    if mask is None:
                        output = tok_loss.mean() / num_microbatches
                    else:
                        mask = mask.flatten(0, 1)
                        tok_loss = tok_loss * mask
                        output = tok_loss.sum() / (mask.sum() + 1e-6) / num_microbatches
                mean_losses.append(output)
                p0_chunk0_batch[1] += 1
            return output

        def deallocate_output_tensor(out):
            """Deallocate the output tensor's '.data' field.
            This method should be called right after the output tensor has been
            sent to the next pipeline stage. At this point, the output tensor is
            only useful for its '.grad_fn' field, and not its '.data'.
            """
            assert isinstance(out, torch.Tensor), (
                "expected Tensor, found %s." % type(out).__name__
            )
            assert out._base is None, "counter-productive to free a view of another tensor."
            out.data.storage().resize_(0)

        def custom_backward(output, grad_output):
            """Custom backward where directly call C++ autograd engine.
            Since Pytorch's 'backward' checks that the output and
            grad have the same shape. We need to manually call the C++ autograd
            instead of using Pytorch's torch.autograd.backward.
            So that the 'deallocate_output_tensor' optimization can work.
            """

            assert (
                output.storage().size() == 0
            ), "output should be pseudo-'freed' in schedule, to optimize memory"
            assert isinstance(output, torch.Tensor), (
                "output == '%s'." % type(output).__name__
            )
            assert isinstance(grad_output, (torch.Tensor, type(None))), (
                "grad_output == '%s'." % type(grad_output).__name__
            )

            # Handle scalar output
            if grad_output is None:
                assert output.numel() == 1, "implicit grad requires scalar output."
                grad_output = torch.ones_like(
                    output,
                    memory_format=torch.preserve_format,
                )

            # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
            Variable._execution_engine.run_backward(
                tensors=(output,),
                grad_tensors=(grad_output,),
                keep_graph=False,
                create_graph=False,
                inputs=tuple(),
                allow_unreachable=True,
                accumulate_grad=True,
            )

        def microbatch_bwd(input_tensor, output_tensor, output_tensor_grad):
            if input_tensor is not None:
                input_tensor.retain_grad()
            if output_tensor_grad is None:
                output_tensor.backward()
            else:
                if args.deallocate_pipeline_outputs:
                    custom_backward(output_tensor, output_tensor_grad)
                else:
                    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
            if input_tensor is not None:
                return input_tensor.grad
            return None

        def forward_step_helper(
            microbatch_id, p0_chunk0_batch, recompute_attn=None, recompute_fc1_fc3=None
        ):
            """Helper method to run forward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            forward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
            mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

            is_first_microbatch = is_first_microbatch_for_model_chunk(microbatch_id)

            # forward step
            if mpu.is_pipeline_first_stage():
                # This is to make sure each model chunk has the number of input same as num_microbatch
                # For other pipeline stages, input will append the received tensor from previous pipeline stage
                if len(input_tensors[model_chunk_id]) == len(
                    output_tensors[model_chunk_id]
                ):
                    input_tensors[model_chunk_id].append(None)

            # input_tensors has all the input for each model chunk.
            # If not first PP stage(including virtual), we will use the very last input in input_tensors.
            # On the first PP stage, if num_microbatch_per_round is larger than pipeline stage,
            # this means we will receive the input num_microbatch_per_round - pipeline_parallel_size earlier than it will be used.
            # So we need to use the input according to index of microbatch. We first figure out in this model chunk, which microbatch we are running.
            # then substract the number of popped input_tensors.
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                input_index = get_input_index(microbatch_id)
                input_index -= len(popped_input_tensors[model_chunk_id])
            else:
                input_index = -1
            input_tensor = input_tensors[model_chunk_id][input_index]
            real_microbatch_id = get_real_microbatch_id(microbatch_id)
            output_tensor = microbatch_fwd(
                model_chunk_id,
                input_tensor,
                microbatch_x[real_microbatch_id],
                microbatch_y[p0_chunk0_batch[1]],
                state,
                (
                    microbatch_mask[real_microbatch_id]
                    if microbatch_mask is not None
                    else None
                ),
                mean_losses,
                is_first_microbatch=is_first_microbatch,
                recompute_attn=recompute_attn,
                recompute_fc1_fc3=recompute_fc1_fc3,
                attn_bias=microbatch_attn_bias[real_microbatch_id],
            )
            output_tensors[model_chunk_id].append(output_tensor)
            return output_tensor

        def backward_step_helper(microbatch_id):
            """Helper method to run backward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            backward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
            mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

            if mpu.is_pipeline_last_stage():
                if len(output_tensor_grads[model_chunk_id]) == 0:
                    output_tensor_grads[model_chunk_id].append(None)
            input_tensor = input_tensors[model_chunk_id].pop(0)
            popped_input_tensors[model_chunk_id].append(None)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

            input_tensor_grad = microbatch_bwd(
                input_tensor, output_tensor, output_tensor_grad
            )
            # Reuse the deallocate_output_tensor function to release input_tensor
            if input_tensor is not None:
                deallocate_output_tensor(input_tensor)

            return input_tensor_grad

        mpu.set_virtual_pipeline_model_parallel_rank(0)
        with record_function("warmup forward passes p2p comm"):
            input_tensors[0].append(
                p2p_communication.recv_forward(
                    tensor_shape, dtype, batch_p2p_comm=batch_p2p_communication
                )
            )

        with record_function("warmup forward passes"):
            fwd_wait_handles = None
            bwd_wait_handles = None
            for k in range(num_warmup_microbatches):
                if fwd_wait_handles is not None:
                    for req in fwd_wait_handles:
                        req.wait()

                # Decide to checkpoint all layers' activations of the current micro-batch
                if max_outstanding_backprops is not None:
                    checkpoint_activations_microbatch = (
                        k % max_outstanding_backprops
                        >= args.num_microbatches_with_partial_activation_checkpoints
                    )
                else:
                    checkpoint_activations_microbatch = None

                with record_function("1f"):
                    output_tensor = forward_step_helper(
                        k,
                        p0_chunk0_batch,
                        recompute_attn=checkpoint_activations_microbatch
                        and args.mb_recompute_attn,
                        recompute_fc1_fc3=checkpoint_activations_microbatch
                        and args.mb_recompute_fc1_fc3,
                    )

                # Determine the model chunk that received input from this iteration belongs to.
                # On the first PP stage, if num_microbatch_per_round is larger than pipeline stage,
                # this means we will receive the input num_microbatch_per_round - pipeline_parallel_size earlier than it will be used by its model chunk.
                # so to determine the true model chunk, we need to add num_microbatch_per_round - pipeline_parallel_size.
                next_forward_model_chunk_id = None
                if mpu.is_pipeline_first_stage(ignore_virtual=True):
                    if microbatch_less_than_pp:
                        next_forward_model_chunk_id = get_model_chunk_id(
                            k + 1,
                            forward=True,
                        )
                    else:
                        next_forward_model_chunk_id = get_model_chunk_id(
                            k + 1 + num_microbatch_per_round - pipeline_parallel_size,
                            forward=True,
                        )
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)


                recv_prev = True
                # For first PP rank, there are two cases that to not receive:
                # (1) Before first model chunk of last PP stage start to run, there is nothing to receive.
                # (2) when last model chunk of last PP stage start running, last PP rank wont send input anymore.
                if mpu.is_pipeline_first_stage(ignore_virtual=True):
                    if microbatch_less_than_pp:
                        if k < num_microbatch_per_round - 1:
                            recv_prev = False
                    else:
                        if k < pipeline_parallel_size - 1:
                            recv_prev = False
                        elif (
                            k
                            >= (num_model_chunks - 1) * num_microbatch_per_round
                            + pipeline_parallel_size
                            - 1
                        ):
                            recv_prev = False
                if k == (total_num_microbatches - 1):
                    recv_prev = False

                # Don't send tensor downstream if on last stage.
                if mpu.is_pipeline_last_stage():
                    output_tensor = None

                # Send and receive tensors as appropriate (send tensors computed
                # in this iteration; receive tensors for next iteration

                (
                    input_tensor,
                    fwd_wait_handles,
                ) = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_communication,
                    overlap_p2p_comm=True,
                )

                if k == (num_warmup_microbatches - 1) and not all_warmup_microbatches:
                    input_tensor_grad = None
                    recv_next = True
                    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    (
                        output_tensor_grad,
                        bwd_wait_handles,
                    ) = p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        batch_p2p_comm=batch_p2p_communication,
                        dtype=dtype,
                        overlap_p2p_comm=True,
                    )

                    output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
                    # make sure number of input tensor is same as number of microbatch
                    if recv_prev:
                        input_tensors[next_forward_model_chunk_id].append(input_tensor)

                if args.deallocate_pipeline_outputs and output_tensor is not None:
                    deallocate_output_tensor(output_tensor)

        # Run 1F1B in steady state.
        with record_function("forward 1F1B steady"):
            for k in range(num_microbatches_remaining):
                # Forward pass.
                forward_k = k + num_warmup_microbatches
                sync_grads = is_last_microbatch_for_model_chunk(k)

                # Decide to checkpoint all layers' activations of the current micro-batch
                if max_outstanding_backprops is not None:
                    checkpoint_activations_microbatch = (
                        forward_k % max_outstanding_backprops
                        >= args.num_microbatches_with_partial_activation_checkpoints
                    )
                else:
                    checkpoint_activations_microbatch = None

                if fwd_wait_handles is not None:
                    for req in fwd_wait_handles:
                        req.wait()

                if args.deallocate_pipeline_outputs and output_tensor is not None:
                    deallocate_output_tensor(output_tensor)
                with record_function("1f"):
                    output_tensor = forward_step_helper(
                        forward_k,
                        p0_chunk0_batch,
                        recompute_attn=checkpoint_activations_microbatch
                        and args.mb_recompute_attn,
                        recompute_fc1_fc3=checkpoint_activations_microbatch
                        and args.mb_recompute_fc1_fc3,
                    )

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

                # Last virtual stage no activation tensor to send
                if mpu.is_pipeline_last_stage():
                    output_tensor = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                if mpu.is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(
                        forward_k - (pipeline_parallel_size - 1), forward=True
                    )
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(
                        forward_k + 1, forward=True
                    )

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                (
                    input_tensor,
                    fwd_wait_handles,
                ) = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_communication,
                    overlap_p2p_comm=True,
                )

                if bwd_wait_handles is not None:
                    for req in bwd_wait_handles:
                        req.wait()

                if input_tensor_grad is not None:
                    deallocate_output_tensor(input_tensor_grad)

                # Backward pass.
                backward_k = k
                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)

                if not args.model.enable_ddp and sync_grads:
                    model[
                        backward_model_chunk_id
                    ].dont_wait_current_stream_for_post_all_gather = True
                with (
                    nullcontext()
                    if sync_grads
                    else model[backward_model_chunk_id].no_sync()
                ):
                    with record_function("1b"):
                        input_tensor_grad = backward_step_helper(backward_k)

                mpu.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

                # First virtual stage no activation gradient tensor to send
                if mpu.is_pipeline_first_stage():
                    input_tensor_grad = None

                # Determine if the current virtual stage has an activation gradient tensor to receive
                recv_next = True
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (pipeline_parallel_size - 1), forward=False
                    )
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k + 1, forward=False
                    )

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_communication,
                    overlap_p2p_comm=True,
                )
                if not args.model.enable_ddp and sync_grads:
                    model[
                        backward_model_chunk_id
                    ].dont_wait_current_stream_for_post_all_gather = True
                with (
                    nullcontext()
                    if sync_grads
                    else model[backward_model_chunk_id].no_sync()
                ):
                    if args.zero_bubble and k >= num_1f1b_microbatches:
                        with record_function("zero bubble 1w"):
                            WeightGradStore.pop()

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.
                if recv_prev:
                    input_tensors[next_forward_model_chunk_id].append(input_tensor)
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        output_tensor_grad
                    )
                model_chunk_id = get_model_chunk_id(backward_k, forward=False)

        if args.deallocate_pipeline_outputs and output_tensor is not None:
            deallocate_output_tensor(output_tensor)

        # Run cooldown backward passes (flush out pipeline).
        with record_function("cooldown backward"):
            if overlap_p2p_communication and bwd_wait_handles is not None:
                for wait_handle in bwd_wait_handles:
                    wait_handle.wait()
                if input_tensor_grad is not None:
                    deallocate_output_tensor(input_tensor_grad)

            if all_warmup_microbatches:
                output_tensor_grads[num_model_chunks - 1].append(
                    p2p_communication.recv_backward(
                        tensor_shape, batch_p2p_comm=batch_p2p_communication, dtype=dtype
                    )
                )
            for k in range(num_microbatches_remaining, total_num_microbatches):
                if overlap_p2p_communication and bwd_wait_handles is not None:
                    for wait_handle in bwd_wait_handles:
                        wait_handle.wait()
                # same as warmup, for last PP stage, currently received grad is
                # (num_microbatch_per_round - pipeline_parallel_size) earlier than its corresponding model chunk
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    if microbatch_less_than_pp:
                        next_backward_model_chunk_id = get_model_chunk_id(
                            k + 1,
                            forward=False,
                        )
                    else:
                        next_backward_model_chunk_id = get_model_chunk_id(
                            k + 1 + num_microbatch_per_round - pipeline_parallel_size,
                            forward=False,
                        )
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
                model_chunk_id = get_model_chunk_id(k, forward=False)
                if not args.model.enable_ddp and is_last_microbatch_for_model_chunk(k):
                    model[
                        model_chunk_id
                    ].dont_wait_current_stream_for_post_all_gather = True
                with (
                    nullcontext()
                    if is_last_microbatch_for_model_chunk(k)
                    else model[model_chunk_id].no_sync()
                ):
                    with record_function("1b"):
                        input_tensor_grad = backward_step_helper(k)

                recv_next = True
                # for last pp stage, if it start the very last model chunk, then no need to receive
                # edge case is when it is bfs, before first model chunk of first pp stage start bwd, last stage doesnt need to receive.
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    if microbatch_less_than_pp:
                        if k < num_microbatch_per_round - 1:
                            recv_next = False
                    else:
                        if k < pipeline_parallel_size - 1:
                            recv_next = False
                        elif (
                            k
                            >= total_num_microbatches
                            - num_microbatch_per_round
                            - 1
                            + pipeline_parallel_size
                        ):
                            recv_next = False
                if k == (total_num_microbatches - 1):
                    recv_next = False

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_communication,
                    overlap_p2p_comm=True,
                )
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        output_tensor_grad
                    )

                with (
                    nullcontext()
                    if is_last_microbatch_for_model_chunk(k)
                    else model[model_chunk_id].no_sync()
                ):
                    with record_function("zero bubble 1w"):
                        WeightGradStore.pop()
        while WeightGradStore.weight_grad_queue.qsize() > 0:
            with record_function("zero bubble 1w"):
                WeightGradStore.pop()

            # Make sure all communication is finished
            torch.cuda.synchronize()

        for model_chunk_id in range(num_model_chunks):
            model[model_chunk_id].dont_wait_current_stream_for_post_all_gather = False
            # logger.warning(f"model_chunk: {model_chunk_id}; rank: {torch.distributed.get_rank()}")
            model[model_chunk_id]._wait_for_post_backward()

        if len(mean_losses) > 0:
            sum_loss_across_mb = torch.stack(mean_losses).sum()
        else:
            sum_loss_across_mb = torch.zeros([], dtype=torch.float32, device="cuda")

        torch.distributed.broadcast(
            sum_loss_across_mb,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        return sum_loss_across_mb, None
