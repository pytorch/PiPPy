# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from pippy.microbatch import (
    merge_chunks,
    split_args_kwargs_into_chunks,
    TensorChunkSpec,
)

from pippy.PipelineStage import PipelineStageBase

logger = logging.getLogger(__name__)

METADATA_TENSOR_LEN = 100
PLACEHOLDER_VAL = -1


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
    If the tensor is None, then this tensor will contain PLACEHOLDER_VALs.

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
    metadata: List[torch.Size] = []
    i = 0
    while i < len(tensor) and tensor[i] != PLACEHOLDER_VAL:
        num_dims = tensor[i].item()
        shape = torch.Size(tensor[i + 1 : i + 1 + num_dims].tolist())
        metadata.append(shape)
        i += num_dims + 1
    return metadata


def get_stage_shapes(
    stage_modules: List[nn.Module],
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
        stage_modules: The chunks assigned to this rank. Rhe length should be 1 for any
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
    for stage_id, model in zip(stage_ids, stage_modules):
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
    logger.info(stage_id_to_shapes)
    return stage_id_to_shapes


class ManualPipelineStage(PipelineStageBase):
    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        device: torch.device,
        num_microbatches: int,
        group: dist.ProcessGroup = None,
        input_args: Optional[Union[torch.Tensor, List[torch.tensor]]] = None,
        output_args: Optional[Union[torch.Tensor, List[torch.tensor]]] = None,
    ):
        super().__init__(
            module, stage_id, num_stages, device, num_microbatches, group
        )
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
        self.inputs_outputs: Deque[Tuple[Tuple[Any, ...], Any]] = deque()

        # these are the buffers used in backwards send/recv, they are allocated later
        self.outputs_grad: List[torch.tensor] = []

        def stage_global_rank(peer_rank):
            return (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )

        self.prev_stage = stage_global_rank((self.rank - 1) % self.world_size)
        self.next_stage = stage_global_rank((self.rank + 1) % self.world_size)

        logger.debug(
            f"""
            finished pipeline stage init, {self.stage_index=}, {self.is_first_stage=},
            {self.is_last_stage=}, {self.num_stages=},
            inputs: {[inp.shape for inp in self.inputs]},
            output: {[output.shape for output in self.outputs]}
            """
        )

    def _retrieve_recv_activations(
        self,
    ):
        """
        Retrieve the activations received for the current stage during forward.
        """
        # TODO: grad always gets set but later
        # we can selectively choose which inputs require grads
        for inp in self.inputs:
            inp.requires_grad_(True)
        return self.inputs

    def _retrieve_recv_grads(
        self,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        """
        # TODO fix so we dont create a tensor
        self.outputs_grad = [torch.empty_like(x) for x in self.outputs]
        return self.outputs_grad

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
            # TODO: cannot split on another dimension other than 0
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self.chunks,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self.chunks, [{}] * self.chunks

    def merge_outputs(self):
        # TODO: manual stage only supports splitting on dimension 0
        # Last rank return merged results per original format
        if self.is_last:
            return merge_chunks(
                self.output_chunks,
                TensorChunkSpec(0),
            )
        else:
            return None

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
            ops.append(
                dist.P2POp(dist.irecv, recv_tensor, self.prev_stage, self.group)
            )
        if not self.is_last_stage:
            ops.append(
                dist.P2POp(dist.isend, send_tensor, self.next_stage, self.group)
            )

        # backward
        if not self.is_first_stage:
            ops.append(
                dist.P2POp(dist.isend, send_tensor, self.prev_stage, self.group)
            )
        if not self.is_last_stage:
            ops.append(
                dist.P2POp(dist.irecv, recv_tensor, self.next_stage, self.group)
            )

        return True

    def get_fwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        return [
            dist.P2POp(dist.irecv, inp, self.prev_stage, self.group)
            for inp in self.inputs
        ]

    def get_fwd_send_ops(self) -> List[dist.P2POp]:
        output_tuples, flatten_input_tensors = self.fwd_cache[
            self.fwd_chunk_id - 1
        ]
        if self.is_last_stage:
            return []
        return [
            dist.P2POp(dist.isend, out, self.next_stage, self.group)
            for out in output_tuples
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

    def get_bwd_recv_ops(self) -> List[dist.P2POp]:
        if self.is_last_stage:
            return []
        return [
            dist.P2POp(dist.irecv, grad, self.next_stage, self.group)
            for grad in self.outputs_grad
        ]

    def get_bwd_send_ops(self) -> List[dist.P2POp]:
        if self.is_first_stage:
            return []
        return [
            dist.P2POp(dist.isend, inp.grad, self.prev_stage, self.group)
            for inp in self.inputs
        ]


def validate_stage_shapes(pipeline_stages: List[ManualPipelineStage]):
    """
    Check that the buffer shapes match between stages was expected by performing an all_gather between
    all stages.
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
        expected_inputs = stage.inputs
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
        expected_outputs = stage.outputs
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
