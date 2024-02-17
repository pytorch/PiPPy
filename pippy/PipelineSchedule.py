# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function

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


def create_buffers(
    tensor: Union[torch.Tensor, List[torch.tensor]], device: torch.device
) -> List[torch.Tensor]:
    """
    Creates buffers for a given tensor on a specified device.
    This function takes as input a tensor or a list of tensors and returns a tensor or a list of tensors (respectively)
    of the same shape, but located on the specified device and uninitialized (i.e., filled with arbitrary data).
    """
    if isinstance(tensor, torch.Tensor):
        return [torch.empty_like(tensor, device=device)]
    elif isinstance(tensor, (list, tuple)):
        return [torch.empty_like(t, device=device) for t in tensor]
    raise TypeError(
        f"Unsupported input type {type(tensor)} cannot create buffers"
    )


METADATA_TENSOR_LEN = 100
PLACEHOLDER_VAL = -1


def create_metadata_tensor(
    tensors: Optional[List[torch.Tensor]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
) -> torch.Tensor:
    """
    Create a metadata tensor that can be sent over the wire.
    This tensor contains the number of dimensions and the shape of each tensor being sent.

    The data is of format [num_dims, dim1, dim2, ...].
    If the tensor is None, a tensor of only placeholder values will be returned.

    Inputs:
        tensors: A list of tensors, the tensors will converted into its shape dimensions and
                 these dimensions will be concatenated.
        device: The device where the metadata tensor will be created.
    If the tensor is None, then this tensor will contain 0s.

    """
    metadata_tensor = torch.full(
        (METADATA_TENSOR_LEN,),
        PLACEHOLDER_VAL,
        dtype=torch.int32,
        device=device,
    )
    if tensors:
        # Create a list of tensors containing the number of dimensions and the shape of each tensor
        data = [
            # data is of format [num_dims, dim1, dim2, ...]
            torch.tensor(
                [len(tensor.shape)] + list(tensor.shape),
                dtype=torch.int32,
                device=device,
            )
            for tensor in tensors
        ]
        # Concatenate the data into a single tensor
        data_tensor = torch.cat(data)
        dt_shape = data_tensor.shape[0]
        if dt_shape > METADATA_TENSOR_LEN:
            raise ValueError(
                f"Metadata tensor size ({dt_shape}) exceeds maximum allowed length ({METADATA_TENSOR_LEN})."
            )
        metadata_tensor[:dt_shape] = data_tensor
    return metadata_tensor


def extract_metadata_from_tensor(tensor: torch.Tensor) -> List[torch.Size]:
    """
    Extract the number of dimensions and the shape of each tensor from a metadata tensor.
    """
    metadata = []
    i = 0
    while i < len(tensor) and tensor[i] != PLACEHOLDER_VAL:
        num_dims = tensor[i].item()
        shape = torch.Size(tensor[i + 1 : i + 1 + num_dims].tolist())
        metadata.append(shape)
        i += num_dims + 1
    return metadata


def get_stage_shapes(
    models: List[nn.Module],
    stage_ids: List[int],
    num_stages: int,
    rank: int,
    world_size: int,
    device: torch.device,
    microbatch: Optional[Union[torch.tensor, List[torch.tensor]]] = None,
):
    """
    Performs a dry run through all the pipeline stages (a rank can have multiple pipeline stages in the case of
    virtual pipelining) and returns the shape of the inputs and outputs of the module.
    Only the first stage must pass in a microbatch.

    Each rank must call get_stage_shapes or the program will hang.

    Args:
        models: The chunks assigned to this rank. Rhe length should be 1 for any
                non-interleaved schedules and >1 for any interleaved schedules.
        stage_ids: The id of the stages assigned to this rank.
        num_stages: Total number of stages.
        rank: Rank of the current process.
        world_size: Number of processes participating in the pipeline.
        device: Device where the tensors are allocated.

    Returns a dictionary containing the following keys:
        "inputs": Shape of the inputs to the module
        "outputs": Shape of the outputs of the module
    """

    stage_id_to_shapes: Dict[int, Dict[str, torch.Size]] = {}
    for stage_id, model in zip(stage_ids, models):
        input_shape_metadata_tensor = create_metadata_tensor(device=device)
        # TODO: Assumes prev_stage == rank - 1 and next_stage == rank + 1
        prev_rank = (rank - 1) % world_size
        next_rank = (rank + 1) % world_size
        shapes = {}

        # first stage doesn't receive anything and uses a microbatch
        if stage_id == 0:
            if microbatch is None:
                raise RuntimeError("Microbatch is required for first stage")
            example_fwd_inputs = microbatch
            if isinstance(example_fwd_inputs, torch.Tensor):
                example_fwd_inputs = [example_fwd_inputs]
        else:
            # other stages must receive shape information
            # TODO: send/recv should take a group, rather than use the default group
            dist.recv(input_shape_metadata_tensor, prev_rank)
            metadata = extract_metadata_from_tensor(input_shape_metadata_tensor)
            example_fwd_inputs = [
                torch.empty(shape_list, device=device)
                for shape_list in metadata
            ]
        shapes["inputs"] = [fwd_input.shape for fwd_input in example_fwd_inputs]

        # perform forward
        # TODO: if forward fails raise a more descriptive error explaining which stage failed
        fwd_outputs = model(*example_fwd_inputs)
        fwd_outputs = create_buffers(fwd_outputs, device)
        shapes["outputs"] = [fwd_output.shape for fwd_output in fwd_outputs]

        # send shape dims
        if stage_id != num_stages - 1:
            output_shape_metadata_tensor = create_metadata_tensor(
                fwd_outputs, device=device
            )
            dist.send(output_shape_metadata_tensor, next_rank)
        stage_id_to_shapes[stage_id] = shapes
    print(stage_id_to_shapes)
    return stage_id_to_shapes


def validate_stage_shapes(pipeline_stages: List[PipelineStageBase]):
    """
    Check that the buffer shapes match between stages was expected by performing an all_gather between
    all stages. Assumes that buffers have been initialized already such that get_fwd_recv_ops() and
    get_fwd_send_ops() return valid lists of p2p ops.
    """
    if len(pipeline_stages) == 0:
        raise ValueError("No pipeline stages provided.")

    virtual_pipeline_size = len(pipeline_stages)
    all_inputs = []
    all_outputs = []
    world_size = pipeline_stages[0].world_size
    num_stages = pipeline_stages[0].num_stages

    # perform all gathers between all stages
    for virtual_id, stage in enumerate(pipeline_stages):
        world_size = stage.world_size
        stage_id = stage.stage_index
        rank = stage.rank
        # check that world_size and num_stages are consistent across all stages
        if stage.world_size != world_size:
            raise ValueError(
                f"Stage id {stage_id} has world size ({stage.world_size}) which does not match world size ({world_size}) of other stages."
            )
        if stage.num_stages != num_stages:
            raise ValueError(
                f"Stage id {stage_id} has num stages ({stage.num_stages}) which does not match num stages ({num_stages}) of other stages."
            )

        # TODO: once we pass in pg to stage, check the pg rank is same as stage rank
        if rank != (pg_rank := dist.get_rank()):
            raise ValueError(
                f"Rank {rank} is not equal to process group rank {pg_rank}"
            )

        if (num_stages := stage.num_stages) % world_size != 0:
            raise ValueError(
                f"Number of stages ({num_stages}) must be a multiple of the world_size ({world_size})"
            )

        # all gather each ranks inputs
        tensor_list = [
            create_metadata_tensor(device=stage.device)
            for _ in range(stage.world_size)
        ]
        expected_inputs = [op.tensor for op in stage.get_fwd_recv_ops()]
        stage_input = create_metadata_tensor(
            expected_inputs, device=stage.device
        )
        dist.all_gather(tensor_list, stage_input)
        stage_input_shapes = [
            extract_metadata_from_tensor(tensor) for tensor in tensor_list
        ]

        # all gather each ranks outputs
        tensor_list = [
            create_metadata_tensor(device=stage.device)
            for _ in range(stage.world_size)
        ]
        expected_outputs = [op.tensor for op in stage.get_fwd_send_ops()]
        stage_output = create_metadata_tensor(
            expected_outputs, device=stage.device
        )
        dist.all_gather(tensor_list, stage_output)
        stage_output_shapes = [
            extract_metadata_from_tensor(tensor) for tensor in tensor_list
        ]

        logger.debug(
            f"""
            Rank: {pg_rank}
            Stage id: {stage_id}
            Stage num stages: {stage.num_stages}
            Stage rank: {rank}
            Stage world size: {world_size}
            Stage {virtual_id * world_size}-{(virtual_id + 1) * world_size - 1} input shapes: {stage_input_shapes}
            Stage {virtual_id * world_size}-{(virtual_id + 1) * world_size - 1} output shapes: {stage_output_shapes}
        """
        )

        all_inputs.extend(stage_input_shapes)
        all_outputs.extend(stage_output_shapes)

    # log only rank 0's view, they will all be equivalent
    if pg_rank == 0:
        logger.info(
            f"""
            all stage inputs: {all_inputs}
            all stage outputs: {all_outputs}
        """
        )

    # Check if the output for stage 0 matches the input at stage 1, and so forth
    for i in range(virtual_pipeline_size * world_size - 1):
        if (out := all_outputs[i]) != (inp := all_inputs[i + 1]):
            raise ValueError(
                f"Stage_id {stage_id} output shape {out} at does not match stage_id {i + 1} input shape {inp}."
            )


class PipelineStageV2Impl(PipelineStageBase):
    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, List[torch.tensor]]] = None,
        output_args: Optional[Union[torch.Tensor, List[torch.tensor]]] = None,
    ):
        super().__init__(stage_id, num_stages, device)
        self.module = module.to(device)
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1
        # When we materialize the model partition on cuda, we call reset_parameters() if it is available
        # logger.info(f"input args {input_args=}")
        self.inputs: List[torch.tensor] = []
        self.outputs: List[torch.tensor] = []

        if input_args is not None:
            self.inputs = create_buffers(input_args, device)

        if output_args is None:
            self.outputs = self.module(*self.inputs)
            # create buffers for the output so that the data is in the correct
            # shape in order to use in p2p op (send)
            self.outputs = create_buffers(self.outputs, device)
        else:
            self.outputs = create_buffers(output_args, device)

        # this is used in backward
        self.inputs_outputs: Deque[
            Tuple[List[torch.tensor], List[torch.tensor]]
        ] = deque()

        # these are the buffers used in backwards send/recv, they are allocated later
        self.inputs_grad: List[torch.tensor] = []
        self.outputs_grad: List[torch.tensor] = []

        # TODO: Calculating stage index from rank is not ideal, e.g. won't match in Interleaved 1F1B case.
        self.prev_stage = (self.rank - 1) % self.world_size
        self.next_stage = (self.rank + 1) % self.world_size

        self.requests: List[dist.P2POp] = []
        logger.debug(
            f"""
            finished pipeline stage init, {self.stage_index=}, {self.is_first_stage=},
            {self.is_last_stage=}, {self.num_stages=},
            inputs: {[inp.shape for inp in self.inputs]},
            output: {[output.shape for output in self.outputs]}
            """
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
            dist.P2POp(dist.irecv, inp, self.prev_stage) for inp in self.inputs
        ]

    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        assert (
            len(self.outputs) != 0
        ), "forward() must be called before get_fwd_send_ops"
        if self.is_last_stage:
            return []
        return [
            dist.P2POp(dist.isend, out, self.next_stage) for out in self.outputs
        ]
        raise AssertionError("invalid fwd_outputs type, cannot send")

    def check_and_format_outputs(self, outputs: Any) -> List[torch.tensor]:
        # validate the output of the module is a type supported
        # supported types: tensor, tuple[torch.tensor], list[torch.tensor]
        if not isinstance(outputs, (torch.Tensor, tuple, list)):
            raise TypeError(
                f"Module output of type {type(outputs)} is not supported"
            )
        if isinstance(outputs, torch.Tensor):
            return [outputs]
        if isinstance(outputs, (tuple, list)) and any(
            not isinstance(x, torch.Tensor) for x in outputs
        ):
            raise TypeError(
                f"All elements in module output must be torch.tensor instances, but got {[type(x) for x in outputs]}"
            )
        return outputs

    def forward_one_chunk(
        self, args: Optional[Union[torch.Tensor, Tuple[torch.tensor]]] = None
    ) -> Any:
        # Non-0 stage
        if args is None:
            args = self.inputs

        # we always expect to unpack a tuple of inputs, so if its a single tensor, wrap it in a tuple
        if isinstance(args, torch.Tensor):
            args = (args,)

        logger.info(
            f"[{self.rank} FORWARD {self.stage_index} {[inp.shape for inp in args]}"
        )

        # this is needed when we access the gradients for this in backward()
        # TODO: requires_grad should not be set, it should depend on input (https://github.com/pytorch/PiPPy/issues/945)
        for tensor in args:
            tensor.requires_grad = True
            tensor.retain_grad()

        # perform forward pass on module
        outputs = self.module(*args)
        self.outputs = self.check_and_format_outputs(outputs)

        # TODO: this is a hack to get the loss, we should be able to get the
        # loss with the loss_fn in PipelineSchedule
        # outputs_or_loss = self.compute_loss() if self.is_last_stage else outputs
        outputs_or_loss = outputs

        # we store a ref to the input/output pair for this forward to be later used by the corresponding backward
        self.inputs_outputs.append((args, outputs_or_loss))

        return outputs_or_loss

    def get_bwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_last_stage:
            return []
        # grads should be same shape as the output from forward()
        self.outputs_grad = [torch.empty_like(out) for out in self.outputs]
        return [
            dist.P2POp(dist.irecv, grad, self.next_stage)
            for grad in self.outputs_grad
        ]

    def get_bwd_send_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        return [
            dist.P2POp(dist.isend, grad, self.prev_stage)
            for grad in self.inputs_grad
        ]

    def backward_one_chunk(self, **kwargs) -> None:
        logger.info(f"[{self.rank} BACKWARD {self.stage_index}]")

        if self.is_last_stage:
            inputs, loss = self.inputs_outputs.popleft()
        else:
            inputs, outputs = self.inputs_outputs.popleft()

        # Compute gradients
        # TODO: HACK materialize_grads=True sets gradients to 0s on backward pass,
        # we need set all the gradients for the inputs that need it, but should not send 0s
        # due to extra communication
        if self.is_last_stage:
            gradients = torch.autograd.grad(
                outputs=loss,
                inputs=inputs,
                retain_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )
        else:
            gradients = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=self.outputs_grad,
                retain_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )

        self.inputs_grad = gradients


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

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                output = self._stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])

                ops = self._stage.get_fwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

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

        # No loss function, no need to run backward
        if not self._has_backward:
            return

        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                loss = internal_losses[i] if len(internal_losses) > 0 else None
                self._stage.backward_one_chunk(loss=loss)

                ops = self._stage.get_bwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

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

        for i in range(total_steps):
            if i < self._n_microbatches:
                # forward
                with record_function(f"Forward {i}"):
                    ops = self._stage.get_fwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    self._stage.forward_one_chunk(arg_mbs[i])

                    ops = self._stage.get_fwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)
            if (
                warmup_steps
                <= i
                < warmup_steps + main_1f1b_steps + cooldown_steps
            ):
                # backward
                with record_function(f"Backward {i}"):
                    ops = self._stage.get_bwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    self._stage.backward_one_chunk()

                    ops = self._stage.get_bwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)

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
