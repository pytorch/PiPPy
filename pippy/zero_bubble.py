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
    microbatch_mask = mask.split(microbatch_size, dim=0)

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

num_model_chunks = len(self._stages)
total_num_microbatches = num_microbatches * num_model_chunks

dtype = torch.fp16 #get_torch_dtype(args.dtype)

#mpu.set_virtual_pipeline_model_parallel_rank(0)
all_warmup_microbatches = False

# if not args.model.enable_ddp:
#     for model_chunk in model:
#         model_chunk._rebuild_full_params_recursive()
# else:
#     for model_chunk in model:
#         model_chunk.zero_grad()

# FSDP only
for model_chunk in self._stages:
    model_chunk._rebuild_full_params_recursive()


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

# max_outstanding_backprops = None
# if args.num_microbatches_with_partial_activation_checkpoints is not None:
#     max_outstanding_backprops = num_warmup_microbatches + 1

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
        assert self.rank == 0 # first stage
    else:
        assert not self.rank != 0

    output, _ = self._stages[model_chunk_id](
        microbatch_tokens,
        pipeline_parallel_input_tensor=input_tensor,
        is_first_microbatch=is_first_microbatch,
        precomputed_attn_bias=attn_bias,
    )

    if self.rank == self.pp_group_size - 1:
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
        # if args.deallocate_pipeline_outputs:
        #     custom_backward(output_tensor, output_tensor_grad)
        # else:
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
    #mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

    is_first_microbatch = is_first_microbatch_for_model_chunk(microbatch_id)

    # forward step
    if self.rank == 0:
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
    if self.rank == 0:
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
    #mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

    if self.rank == self.pp_group_size -1:
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

#mpu.set_virtual_pipeline_model_parallel_rank(0)
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
        # if max_outstanding_backprops is not None:
        #     checkpoint_activations_microbatch = (
        #         k % max_outstanding_backprops
        #         >= args.num_microbatches_with_partial_activation_checkpoints
        #     )
        # else:
        checkpoint_activations_microbatch = None

        with record_function("1f"):
            output_tensor = forward_step_helper(
                k,
                p0_chunk0_batch,
                recompute_attn=checkpoint_activations_microbatch,
                recompute_fc1_fc3=checkpoint_activations_microbatch
            )

        # Determine the model chunk that received input from this iteration belongs to.
        # On the first PP stage, if num_microbatch_per_round is larger than pipeline stage,
        # this means we will receive the input num_microbatch_per_round - pipeline_parallel_size earlier than it will be used by its model chunk.
        # so to determine the true model chunk, we need to add num_microbatch_per_round - pipeline_parallel_size.
        next_forward_model_chunk_id = None
        if self.rank == 0:
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
        if self.rank == 0:
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
        if self.rank == self.pp_group_size - 1:
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
            if self.rank == self.pp_group_size - 1:
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

        if self.deallocate_pipeline_outputs and output_tensor is not None:
            deallocate_output_tensor(output_tensor)

# Run 1F1B in steady state.
with record_function("forward 1F1B steady"):
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        sync_grads = is_last_microbatch_for_model_chunk(k)

        # Decide to checkpoint all layers' activations of the current micro-batch
        # if max_outstanding_backprops is not None:
        #     checkpoint_activations_microbatch = (
        #         forward_k % max_outstanding_backprops
        #         >= args.num_microbatches_with_partial_activation_checkpoints
        #     )
        # else:
        checkpoint_activations_microbatch = None

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        if self.deallocate_pipeline_outputs and output_tensor is not None:
            deallocate_output_tensor(output_tensor)
        with record_function("1f"):
            output_tensor = forward_step_helper(
                forward_k,
                p0_chunk0_batch,
                recompute_attn=checkpoint_activations_microbatch,
                recompute_fc1_fc3=checkpoint_activations_microbatch,
            )

        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        #mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

        # Last virtual stage no activation tensor to send
        if self.rank == self.pp_group_size - 1:
            output_tensor = None

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        if self.rank == 0:
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
