# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.utils._pytree import tree_flatten, tree_unflatten


class CustomReducer:
    def __init__(self, init_value, reduce_fn):
        self.init_value = init_value
        self.reduce_fn = reduce_fn

class TensorChunkSpec:
    def __init__(self, split_dim):
        self.split_dim = split_dim

    split_dim : int

    def __repr__(self):
        return f'{self.__class__.__module__}.{self.__class__.__name__}({self.split_dim})'

    def __str__(self):
        return f'TensorChunkSpec({self.split_dim})'

def shard_dict_of_args(args_dict, args_chunk_spec, num_chunks, _debug_mask_minibatches : bool = False):
    # Stage 1+2: flatten and shard/replicate

    # args_sharded_replicated : [num args, num flat values, num chunks]
    args_sharded_replicated = {}

    arg_specs = []

    assert len(args_dict) == len(args_chunk_spec)
    for arg_key, arg in args_dict.items():
        chunk_spec = args_chunk_spec[arg_key]
        flat, spec = tree_flatten(arg)
        arg_specs.append(spec)
        chunk_spec_flat, _ = tree_flatten(chunk_spec)

        if len(flat) != len(chunk_spec_flat):
            raise ValueError(f'Argument value {arg} did not have the same number of '
                             f'values as as chunk spec {chunk_spec}')

        sharded_arg_flat = []

        for v, chunk_v in zip(flat, chunk_spec_flat):
            if isinstance(chunk_v, TensorChunkSpec):
                # TODO: check type of v. If it's a tensor, use chunk (or debug mask).
                # If it's a collection type, split it as you would expect. Otherwise,
                # Throw an error
                assert isinstance(v, torch.Tensor)

                chunk_tensors = torch.tensor_split(v, num_chunks, chunk_v.split_dim)

                if _debug_mask_minibatches:
                    expanded_chunks = []

                    split_dim_idx = 0
                    for chunk_tensor in chunk_tensors:
                        new_val = torch.zeros_like(v)
                        upper_idx = split_dim_idx + chunk_tensor.size(chunk_v.split_dim)

                        slice_indices = [slice(None, None, None)] * new_val.ndim
                        slice_indices[chunk_v.split_dim] = slice(split_dim_idx, upper_idx)
                        new_val[slice_indices] = chunk_tensor

                        expanded_chunks.append(new_val)

                        split_dim_idx += chunk_tensor.size(chunk_v.split_dim)

                    sharded_arg_flat.append(expanded_chunks)
                else:
                    sharded_arg_flat.append(chunk_tensors)
            else:
                sharded_arg_flat.append([v] * num_chunks)

        args_sharded_replicated[arg_key] = sharded_arg_flat

    # chunks_flat : [num chunks, num args, num flat values]
    chunks_flat = []
    for chunk_idx in range(num_chunks):
        chunk_args = {}
        for key, arg in args_sharded_replicated.items():
            arg_single_chunk = []
            for v_flat in arg:
                arg_single_chunk.append(v_flat[chunk_idx])
            chunk_args[key] = arg_single_chunk
        chunks_flat.append(chunk_args)


    # args_split : [num chunks, num args]
    args_split = []

    for chunk in chunks_flat:
        per_chunk_args = {}
        assert len(arg_specs) == len(chunk)
        for (key, arg), arg_spec in zip(chunk.items(), arg_specs):
            per_chunk_args[key] = tree_unflatten(arg, arg_spec)
        args_split.append(per_chunk_args)

    return args_split


def split_args_kwargs_into_chunks(args, kwargs, args_chunk_spec, kwargs_chunk_spec, chunks, 
                                  _debug_mask_minibatches : bool = False):
    # Given `args` and `kwargs`, we want to yield a set of `chunks` args and kwargs such that
    # the constituent Tensor values have been sharded/replicated according to the `args_chunk_spec`
    # and `kwargs_chunk_spec` specifications. The steps are as follows:
    #
    # 1. Use pytree.tree_flatten to flatten each arg and its spec into nto a 1d array of values.
    #    To use a running example: suppose our inputs look like
    #
    #       args = ([A, [B, C]], D) args_spec = ([None, [None, TensorChunkSpec]], None)
    #       (kwargs not shown but it's a similar process)
    #
    #    Then for this step we would end up with
    #
    #       args = ([A, B, C], D) args_spec = ([None, None, TensorChunkSpec], None)
    #
    # 2. Shard or replicate the arguments subject to the policy in the spec. Suppose chunks = 2
    #
    #       args = ([[A, A], [B, B], [C_1, C_2]], [D, D])
    #
    # 3. Rotate the nesting order such that chunks are the outer dimension
    #
    #       args_chunks = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 4. Unflatten each chunk according to the spec
    #
    #       args_chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]

    # TODO: _debug_mask_minibatches

    args_split_dict = shard_dict_of_args(
        dict(enumerate(args)), dict(enumerate(args_chunk_spec)), chunks, _debug_mask_minibatches)
    kwargs_split = shard_dict_of_args(kwargs, kwargs_chunk_spec, chunks, _debug_mask_minibatches)

    args_split = []
    for chunk_args in args_split_dict:
        args_split.append(tuple(chunk_args[i] for i in range(len(chunk_args))))

    return args_split, kwargs_split


def merge_chunks(chunks, chunk_spec, _debug_mask_minibatches : bool = False):
    # Given a list of chunks and a chunk specification, merge the chunks
    # into a single value according to that chunk spec. This is essentially
    # the inverse of `split_args_kwargs_into_chunks`, so the steps are
    # similar to the steps in that function but in reverse. Given the
    # input values:
    #
    #       chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]
    #       args_spec = ([None, [None, TensorChunkSpec]], None)
    #
    # 1. Flatten the chunks according to the chunk_spec
    #
    #       chunks_flat = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 2. Rotate the nesting order such that chunks are the inner dimension
    #
    #       value_inner = ([A, B, [C_1, C_2]], D)
    #
    # 3. Concatenate sharded arguments
    #
    #       value_combined = ([A, B, C], D)
    #
    # 4. Unflatten the combined args given the spec
    #
    #       value = ([A, [B, C]], D)

    # Preliminary: flatten the chunk spec
    spec_flattened, flatten_spec = tree_flatten(chunk_spec)

    # Stage 1: flatten chunks
    # chunks_flattened : [num chunks, num args]
    chunks_flattened = []

    for chunk in chunks:
        chunk_flattened, _ = tree_flatten(chunk)
        if len(chunk_flattened) != len(spec_flattened):
            raise ValueError(f'Chunk {chunk} did not match chunk spec {chunk_spec}')

        chunks_flattened.append(chunk_flattened)

    # Stage 2 and 3: Rotate nesting order s.t. chunks are inner dimension and
    #                concatenate sharded operands
    # args_flattened : [num args]
    args_flattened = []
    for arg_idx, arg in enumerate(spec_flattened):
        if isinstance(arg, TensorChunkSpec):
            partial_values = [chunks_flattened[chunk_idx][arg_idx] for chunk_idx in range(len(chunks_flattened))]

            if _debug_mask_minibatches:
                # Infer size of individual chunks by running `tensor_split` again
                overall_shape = partial_values[0].shape
                for val in partial_values[1:]:
                    assert val.shape == overall_shape
                meta_chunks = torch.tensor_split(
                    torch.empty(*overall_shape, device='meta'), sections=len(partial_values), dim=arg.split_dim)

                values_to_cat = []
                chunk_start_idx = 0
                assert len(partial_values) == len(meta_chunks)
                for partial_value, meta_chunk in zip(partial_values, meta_chunks):
                    chunk_end_idx = chunk_start_idx + meta_chunk.size(arg.split_dim)

                    slice_indices = [slice(None, None, None)] * partial_value.ndim
                    slice_indices[arg.split_dim] = slice(chunk_start_idx, chunk_end_idx)
                    sliced = partial_value[slice_indices]
                    values_to_cat.append(sliced)

                    chunk_start_idx = chunk_end_idx

            else:
                values_to_cat = partial_values

            args_flattened.append(torch.cat(values_to_cat, dim=arg.split_dim))
        elif isinstance(arg, CustomReducer):
            reduced_val = arg.init_value

            for chunk_idx in range(len(chunks_flattened)):
                reduced_val = arg.reduce_fn(reduced_val, chunks_flattened[chunk_idx][arg_idx])

            args_flattened.append(reduced_val)
        else:
            value = chunks_flattened[0][arg_idx]
            for chunk_idx in range(1, len(chunks_flattened)):
                assert chunks_flattened[chunk_idx][arg_idx] == value
            args_flattened.append(value)

    # Stage 4: Unflatten combined args
    return tree_unflatten(args_flattened, flatten_spec)
