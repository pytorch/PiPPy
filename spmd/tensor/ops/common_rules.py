# Copyright (c) Meta Platforms, Inc. and affiliates
import math

import torch
from typing import List, Sequence, Dict, Tuple, Optional, cast
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.placement_types import DTensorSpec
from spmd.tensor.ops.utils import as_list, normalize_dims


def _replace_char_in_str(string: str, new_char: str, idx: int) -> str:
    return string[:idx] + new_char + string[idx + 1 :]


def _inplace_rewrap_schema_suggestion(
    suggestion: OpSchema, input_schema: OpSchema
) -> None:
    suggestion_args_spec = suggestion.args_spec
    new_arg_schema: List[object] = []
    idx_of_args_spec = 0
    for arg in input_schema.args_schema:
        if isinstance(arg, DTensorSpec):
            new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
            idx_of_args_spec += 1
        else:
            new_arg_schema.append(arg)
    suggestion.args_schema = tuple(new_arg_schema)
    suggestion.kwargs_schema = input_schema.kwargs_schema


def _gen_reshard_suggestions(
    op_schema: OpSchema,
    input_dims: List[str],
    input_specs: Tuple[DTensorSpec, ...],
    dim_to_sharding: Dict[str, int],
    pending_sum: List[int],
) -> OutputSharding:
    suggested_arg_specs: List[DTensorSpec] = []
    for input_dim, input_spec in zip(input_dims, input_specs):
        dim_map = [dim_to_sharding[dim] for dim in input_dim]
        suggested_arg_specs.append(
            DTensorSpec.from_dim_map(
                mesh=input_spec.mesh,
                dim_map=dim_map,
                sums=pending_sum,
                shape=input_spec.shape,
            )
        )
    return OutputSharding(
        None,
        schema_suggestions=[
            OpSchema(op_schema.func_schema, tuple(suggested_arg_specs), {})
        ],
        failed_reason="Input placements op sharding propagation failed, need to reshard!",
    )


def einop_rule(
    equation: str,
    op_schema: OpSchema,
    *,
    linearity: bool = False,
    enforce_sharding: Optional[Dict[str, int]] = None,
) -> OutputSharding:
    """
    Propagate the sharding of inputs to output for ops whose data
    moves according to einsum notation. This is mostly borrowed
    from @zdevito's sharding simulator. Examples:
        mk,kn->mn - einsum
        ij,ij->ij - addition
        ij,j->ij - broadcasted addition
        ij->i - reduction
    Other ops could use this propagation algorithm when applied, note
    that einsum propagation only deal with list of specs (DTensor specs)
    as it only works on list of tensors!
    """
    # parse einop equation and extract arg specs
    inputs, outputs = equation.split("->")
    input_dims, output_dims = inputs.split(","), outputs.split(",")
    input_specs = op_schema.args_spec
    # NOTE: only support single output unless needed in future
    output_dim = output_dims[0]

    dim_to_sharding: Dict[str, int] = {}
    dim_to_size: Dict[str, int] = {}
    # record pending sum, key is mesh dimension, value is pending sum
    # counter across input specs
    pending_sums_counter: Dict[int, int] = {}
    seen_shardings: Dict[int, str] = {}
    needs_reshard = False

    def merge_sharding(dim: str, a: int, b: int) -> int:
        # merge the sharding of inputs if it's able to merge, i.e. we can merge
        # replicate and shard to shard, but this will trigger an reshard operation
        if a != b:
            if a == -1 or b == -1:
                # reshard the replicate to match the sharded one
                nonlocal needs_reshard
                needs_reshard = True
                return a if a != -1 else b
            else:
                # TODO: further merge the sharding properly (i.e. reshard one input to replicate)
                raise RuntimeError(
                    f"{equation}: dim {dim} sharded two different ways: {a} and {b}"
                )
        else:
            return a

    for input_dim, input_spec in zip(input_dims, input_specs):
        # deal with partial sums
        input_sums = input_spec.sums
        for sum_dim in input_sums:
            if sum_dim not in pending_sums_counter:
                seen_shardings[sum_dim] = "+"
            # update pending sum counter for pending sum mesh
            # dimension with the occurance from each input
            pending_sums_counter[sum_dim] = (
                pending_sums_counter.get(sum_dim, 0) + 1
            )

        for idx, (dim, mesh_dim) in enumerate(
            zip(input_dim, input_spec.dim_map)
        ):
            if enforce_sharding and dim in enforce_sharding:
                if enforce_sharding[dim] != mesh_dim:
                    needs_reshard = True
                dim_to_sharding[dim] = enforce_sharding[dim]
                dim_to_size[dim] = input_spec.shape[idx]
            elif dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
                dim_to_size[dim] = input_spec.shape[idx]
            else:
                dim_to_sharding[dim] = merge_sharding(
                    dim, dim_to_sharding[dim], mesh_dim
                )
                assert dim_to_size[dim] == input_spec.shape[idx]

            # after merging sharding, we check if there're multiple
            # sharding on the same mesh dim.
            merged_sharding_for_dim = dim_to_sharding[dim]
            if merged_sharding_for_dim != -1:
                if (
                    merged_sharding_for_dim in seen_shardings
                    and dim != seen_shardings[merged_sharding_for_dim]
                ):
                    needs_reshard = True
                    seen_shardings[merged_sharding_for_dim] += dim
                else:
                    seen_shardings[merged_sharding_for_dim] = dim

    if pending_sums_counter and not linearity:
        # return reshard suggestion with no pending sum, because we already properly
        # merge the sharding, this reshard suggestion is legit to use
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, []
        )
    else:
        # It's a op that support linearity, but not all input arguments are partial
        # we fail the sharding propagation with suggestion to make all inputs be
        # partial on the corresponding mesh dim (all inputs should be partial for
        # the mesh dims in order to execute locally and delay the sum reduction)
        for value in pending_sums_counter.values():
            if value != len(input_specs):
                needs_reshard = True

    for mesh_dim, dims in seen_shardings.items():
        if len(dims) > 1:
            # we found different input dims are being sharded on the same mesh dim
            # in order to perform local op computation, we need to reshard inputs
            # base on some simple heuristics, now we simply pick the one with least comm
            # volume. (i.e. the input with least size)
            # TODO: consider a more advanced heuristic to pick the best sharding
            costs = []
            for d in dims:
                cost = 0
                for input_dim, input_spec in zip(input_dims, input_specs):
                    if (
                        d in input_dim
                        and input_spec.dim_map[input_dim.index(d)] == mesh_dim
                    ):
                        cost += math.prod(
                            input_spec.local_shape
                        ) * input_spec.mesh.size(mesh_dim)
                costs.append(cost)
            d_to_keep_sharding = dims[costs.index(max(costs))]
            for d in dims:
                # update dim_to_sharding to keep the sharding of the dim with
                # highest comm and make the rest of the dims to replicate
                if d != d_to_keep_sharding:
                    dim_to_sharding[d] = -1

    pending_sums = list(pending_sums_counter.keys())
    if needs_reshard:
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, pending_sums
        )

    # if no need to reshard, we directly generate the output sharding
    for dim, shard_on_mesh in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)

    output_dim_map = []
    output_shape = []
    for dim in output_dim:
        if dim == "1":
            # find output dim that is a singleton dimension, mark sharding and shape
            output_dim_map.append(-1)
            output_shape.append(1)
        else:
            output_dim_map.append(dim_to_sharding[dim])
            output_shape.append(dim_to_size[dim])

    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_specs[0].mesh,
            output_dim_map,
            pending_sums,
            shape=torch.Size(output_shape),
        )
    )


def pointwise_rule(
    op_schema: OpSchema, linearity: bool = False
) -> OutputSharding:
    """
    Propagate the sharding for pointwise operations. Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # find the max_dim first in case we need to broadcasting
    input_specs = op_schema.args_spec
    max_dim = max(input.ndim for input in input_specs)
    dimchars = []
    dimchar_singleton_counter: Dict[str, int] = {}
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        # handle the "broadcasting to a common shape case"
        # see https://pytorch.org/docs/stable/notes/broadcasting.html
        # If any of the dimensions is singleton dimension (i.e. 1).
        # we mark the dim char as a special "1" to distinguish with
        # the non-singleton dimension, so that sharding propagation
        # should just ignore the singleton dimension.
        if len(input_specs) > 1:
            for i, dim_length in enumerate(input.shape):
                if dim_length == 1:
                    # mark singleton dim char as a special "1" in einop rule
                    dimchar_singleton_counter[p[i]] = (
                        dimchar_singleton_counter.get(p[i], 0) + 1
                    )
                    p = _replace_char_in_str(p, "1", i)
        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    # check if we replace the all inputs dim char with singleton dimension,
    # if we replace all inputs, we also need to replace the output dimension.
    for output_dim_idx in range(len(out_dimchars)):
        out_dimchar = out_dimchars[output_dim_idx]
        if dimchar_singleton_counter.get(out_dimchar, 0) == len(input_specs):
            out_dimchars = _replace_char_in_str(
                out_dimchars, "1", output_dim_idx
            )

    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"
    einop_schema = OpSchema(op_schema.func_schema, input_specs, {})

    enforce_sharding: Dict[str, int] = {}
    if op_schema.is_inplace:
        # inplace op should keep the input sharding it writes to
        for out_dimchar, mesh_dim in zip(out_dimchars, input_specs[0].dim_map):
            enforce_sharding[out_dimchar] = mesh_dim

    elif op_schema.is_out_variant:
        out_spec = cast(DTensorSpec, op_schema.kwargs_schema["out"])
        for out_dimchar, mesh_dim in zip(out_dimchars, out_spec.dim_map):
            enforce_sharding[out_dimchar] = mesh_dim

    output_sharding = einop_rule(
        fmt,
        einop_schema,
        linearity=linearity,
        enforce_sharding=enforce_sharding,
    )
    if (
        output_sharding.output_spec is None
        and output_sharding.schema_suggestions is not None
    ):
        # sharding propagation failed, with reshard suggetion only have tensor specs,
        # we will need to include back all non-tensor args/kwargs
        suggested_schema = output_sharding.schema_suggestions[0]
        _inplace_rewrap_schema_suggestion(suggested_schema, op_schema)

    return output_sharding


def linear_pointwise_rule(op_schema: OpSchema) -> OutputSharding:
    """
    Linear pointwise operators can propagate pending reductions.
    For example, c = add(a, b); if a is pending sum, then c will be
    pending sum as well without any communication overhead.
    """
    return pointwise_rule(op_schema, linearity=True)


def reduction_rule(op_schema: OpSchema) -> OutputSharding:
    """
    Propagate the sharding for reduction operations. Examples:
        ij->i - sum on dim
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # reduction op usually begin with a single tensor
    input_spec = cast(DTensorSpec, op_schema.args_schema[0])
    input_chars = alphabet[: input_spec.ndim]

    if len(op_schema.args_schema) == 1:
        # reducing to a single scalar tensor, we just mark output as empty
        out_dimchars = ""
    else:
        # this is usually a dim-based reduction op schema pattern, it
        # might not be true for every op for other special cases, we
        # need to specialize them as needed.
        # TODO: add support for things like `torch.unique` where it
        # does not follow the reduction op convention.
        keep_dim = len(op_schema.args_schema) > 2 and bool(
            op_schema.args_schema[2]
        )
        dim_list = cast(Sequence[int], as_list(op_schema.args_schema[1]))
        dim_list = normalize_dims(dim_list, input_spec.ndim)
        # keep_dim=True means output dim is a singleton dim
        reduce_dim_char = ord("1") if keep_dim else None
        out_dimchars = input_chars.translate(
            {ord(alphabet[dim]): reduce_dim_char for dim in dim_list}
        )

    fmt = f"{input_chars}->{out_dimchars}"
    return einop_rule(fmt, op_schema)
