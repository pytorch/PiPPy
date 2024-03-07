# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)


class PipelineStageBase(ABC, nn.Module):
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

    @abstractmethod
    def compute_loss(self):
        """
        Compute loss from the outputs of the last stage
        """
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
    for stage_id, model in zip(stage_ids, models, strict=True):
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
    virtual_pipeline_size = len(pipeline_stages)
    all_inputs = []
    all_outputs = []
    # perform all gathers between all stages
    for virtual_id, stage in enumerate(pipeline_stages):
        world_size = stage.world_size
        stage_id = stage.stage_id
        rank = stage.rank

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
        rank: int,
        world_size: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, List[torch.tensor]]] = None,
        output_args: Optional[Union[torch.Tensor, List[torch.tensor]]] = None,
    ):
        super().__init__()
        self.module = module.to(device)
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.rank = rank
        self.world_size = world_size
        self.device = device
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

        self.prev_stage = (rank - 1) % world_size
        self.next_stage = (rank + 1) % world_size

        self.requests: List[dist.P2POp] = []
        logger.debug(
            f"""
            finished pipeline stage init, {self.stage_id=}, {self.is_first_stage=},
            {self.is_last_stage=}, {self.num_stages=},
            {[inp.shape for inp in self.inputs]},
            {[output.shape for output in self.outputs]}
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
        self, args: Union[torch.Tensor, List[torch.tensor]]
    ) -> Any:
        # we always expect to unpack a tuple of inputs, so if its a single tensor, wrap it in a tuple
        if isinstance(args, torch.Tensor):
            args = (args,)

        logger.info(
            f"[{self.rank} FORWARD {self.stage_id} {[inp.shape for inp in args]}"
        )

        # this is needed when we access the gradients for this in backward()
        # TODO: requires_grad should not be set, it should depend on input (https://github.com/pytorch/PiPPy/issues/945)
        for tensor in args:
            tensor.requires_grad = True
            tensor.retain_grad()

        # perform forward pass on module
        outputs = self.module(*args)
        self.outputs = self.check_and_format_outputs(outputs)

        outputs_or_loss = self.compute_loss() if self.is_last_stage else outputs

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
        logger.info(f"[{self.rank} BACKWARD {self.stage_id}]")

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

    def compute_loss(self):
        if self.outputs is None:
            raise RuntimeError("forward() must be called before compute_loss()")
        # TODO: use a real loss function passed in
        return self.outputs[0].mean()


class PipelineSchedule(ABC):
    def __init__(self, stage: PipelineStageBase, n_microbatches: int):
        self._stage = stage
        self._n_microbatches = n_microbatches

    @abstractmethod
    def step(self, microbatches: Optional[List]=None) -> None:
        """
        Run one iteration of the pipeline schedule. Will go through all the microbatches
        according to the schedule implementation.

        Args:
            microbatches: list of microbatch tensors
        """
        raise NotImplementedError


class PipelineScheduleGPipe(PipelineSchedule):
    def step(self, microbatches: Optional[List]=None):
        if microbatches is not None:
            assert len(microbatches) == self._n_microbatches

        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                if microbatches is not None:
                    self._stage.forward_one_chunk(microbatches[i])
                else:
                    self._stage.forward_one_chunk(self.inputs)

                ops = self._stage.get_fwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

        for i, _ in enumerate(microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops()
                if ops:
                    dist.batch_isend_irecv(ops).pop().wait()

                self._stage.backward_one_chunk(chunk=i)

                ops = self._stage.get_bwd_send_ops()
                if ops:
                    dist.batch_isend_irecv(ops)

            logger.info(f"{self._stage.stage_id} backward mb {i} finished")


class PipelineScheduleLoopedBFS(PipelineSchedule):
    def __init__(self, stages: List[PipelineStageBase]):
        self._stages = stages

    def step(self, microbatches):
        for s, stage in enumerate(self._stages):
            for i, mb in enumerate(microbatches):
                with record_function(f"Stage {s} Forward"):
                    ops = stage.get_fwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    stage.forward_one_chunk(mb)

                    ops = stage.get_fwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)

        for stage in reversed(self._stages):
            for i in range(len(microbatches)):
                with record_function(f"Stage {stage.stage_id} Backward"):
                    ops = stage.get_bwd_recv_ops()
                    if ops:
                        dist.batch_isend_irecv(ops).pop().wait()

                    stage.backward_one_chunk(chunk=i)

                    ops = stage.get_bwd_send_ops()
                    if ops:
                        dist.batch_isend_irecv(ops)


class PipelineScheduleLoopedDFS(PipelineSchedule):
    def __init__(
        self, stages: List[PipelineStageBase], n_microbatch, pp_id, n_pp
    ):
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
        logger.info(
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
        logger.info(
            f"rank {self.pp_id} - minibatch_index {[minibatch_index(step) for step in range(self.total_steps)]}"
        )
        logger.info(
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
                    logger.info(
                        f"rank {self.pp_id} - forward edge case for first stage"
                    )
                    dist.batch_isend_irecv(forward_first_recv).pop().wait()
                    forward_first_recv = None

                if forward_batched_op_handle:
                    logger.info(
                        f"rank: {self.pp_id} - waiting on batched_op_handle before fwd"
                    )
                    forward_batched_op_handle.wait()
                    forward_batched_op_handle = None

                with record_function(f"Stage {forward_stage.stage_id} Forward"):
                    logger.info(
                        f"pp_id {self.pp_id} step {step} forward_stage {forward_stage.stage_id} mb_id {mb_id_fwd}"
                    )
                    forward_stage.forward_one_chunk(microbatches[mb_id_fwd])

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
                    logger.info(
                        f"rank: {self.pp_id}, current stage_id {self.stage_id_to_global_stage_id[fwd_stage_id]}, - {[(req.op, req.peer) for req in requests]}"
                    )
                    forward_batched_op_handle = dist.batch_isend_irecv(
                        requests
                    ).pop()

            if step >= self.warmup_steps:
                if backward_first_recv:
                    logger.info(
                        f"rank {self.pp_id} - backward edge case for last stage"
                    )
                    dist.batch_isend_irecv(backward_first_recv).pop().wait()
                    backward_first_recv = None

                if backward_batched_op_handle:
                    logger.info(
                        f"rank: {self.pp_id} - waiting on batched_op_handles before bwd"
                    )
                    backward_batched_op_handle.wait()
                    backward_batched_op_handle = None

                with record_function(
                    f"Stage {backward_stage.stage_id} Backward"
                ):
                    logger.info(
                        f"pp_id {self.pp_id} step {step}/{self.total_steps} backward_step {backward_step} backward_stage_id {backward_stage.stage_id} mb_id {mb_id_bwd}"
                    )
                    backward_stage.backward_one_chunk(chunk=mb_id_bwd)

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
                    logger.info(
                        f"rank: {self.pp_id} - {[(req.op, req.peer) for req in requests]}"
                    )
                    backward_batched_op_handle = dist.batch_isend_irecv(
                        requests
                    ).pop()

        logger.info("Step exiting")
