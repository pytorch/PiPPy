# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.profiler import record_function

from .IR import Pipe
from .microbatch import merge_chunks, split_args_kwargs_into_chunks
from .PipelineStage import PipelineStageBase

logger = logging.getLogger(__name__)


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
    def step_microbatches(
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
            assert isinstance(
                losses, list
            ), f"losses must be a list but got a {type(losses)}"

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
    Derived classes should implement `step_microbatches`.
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
        self.step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        if self._stage.is_last:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None


class ScheduleGPipe(PipelineScheduleSingle):
    def step_microbatches(
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
    def step_microbatches(
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
    Derived classes should implement `step_microbatches`.
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
        self.step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        for stage in self._stages:
            if stage.is_last:
                return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage
        return None


class ScheduleLoopedBFS(PipelineScheduleMulti):
    def step_microbatches(
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

    def step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
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
        arg_mbs, kwarg_mbs = self._check_inputs(
            arg_mbs, kwarg_mbs, target_mbs, losses
        )

        # warmup steps for latest pp stage is trivial to compute
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
        self.total_steps = warmup_steps + fwd_bwd_steps + cooldown_steps

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

        for step in range(self.total_steps):
            # warmup, forward only
            if step < warmup_steps:
                logger.debug(f"{forward_stage_local_index(step)=}")

                fwd_stage = self._stages[forward_stage_local_index(step)]
                # assigns the current microbatch index and updates it for future steps
                fwd_stage_mb_index[fwd_stage] = (
                    mb_index := fwd_stage_mb_index[fwd_stage]
                ) + 1

                logger.debug(
                    f"{self.rank}: {step=}, {fwd_stage.stage_index=}, {mb_index=}"
                )

                with record_function(f"Forward {step}"):
                    ops = fwd_stage.get_fwd_recv_ops()
                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    output = fwd_stage.forward_one_chunk(arg_mbs[mb_index], kwarg_mbs[mb_index])  # type: ignore[index]

                    ops = fwd_stage.get_fwd_send_ops()
                    works = sorted_batch_isend_irecv(ops)
                    sends_to_wait.extend(works.values())

                    self._maybe_compute_loss(
                        fwd_stage, output, target_mbs, mb_index
                    )

            # 1f1b
            elif warmup_steps <= step < warmup_steps + fwd_bwd_steps:
                logger.debug(f"{forward_stage_local_index(step)=}")
                logger.debug(f"{backward_stage_local_index(step)=}")

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
                    f"{self.rank}: {step=}, {fwd_stage.stage_index=}, {bwd_stage.stage_index=}, {fwd_mb_index=}, {bwd_mb_index=}"
                )
                with record_function(f"1F1B {step}"):
                    ops = fwd_stage.get_fwd_recv_ops()
                    ops.extend(bwd_stage.get_bwd_recv_ops())
                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    # fwd
                    output = fwd_stage.forward_one_chunk(arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]
                    ops = fwd_stage.get_fwd_send_ops()
                    self._maybe_compute_loss(
                        fwd_stage, output, target_mbs, fwd_mb_index
                    )

                    # bwd
                    loss = self._maybe_get_loss(bwd_stage, bwd_mb_index)
                    bwd_stage.backward_one_chunk(loss=loss)
                    ops.extend(bwd_stage.get_bwd_send_ops())

                    works = sorted_batch_isend_irecv(ops)
                    sends_to_wait.extend(works.values())

            # cooldown
            else:
                bwd_stage = self._stages[backward_stage_local_index(step)]
                bwd_stage_mb_index[bwd_stage] = (
                    bwd_mb_index := bwd_stage_mb_index[bwd_stage]
                ) + 1
                bwd_stage._configure_data_parallel_mode(
                    bwd_mb_index == self._n_microbatches - 1
                )
                logger.debug(
                    f"{self.rank}: {step=}, {bwd_stage.stage_index=}, {bwd_mb_index=}"
                )
                with record_function(f"Cooldown (backward) {step}"):
                    ops = bwd_stage.get_bwd_recv_ops()
                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    loss = self._maybe_get_loss(bwd_stage, bwd_mb_index)
                    bwd_stage.backward_one_chunk(loss=loss)

                    ops = bwd_stage.get_bwd_send_ops()
                    works = sorted_batch_isend_irecv(ops)
                    sends_to_wait.extend(works.values())

        # Make sure all sends are finished
        for work in sends_to_wait:
            work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)
