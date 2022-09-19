# Copyright (c) Meta Platforms, Inc. and affiliates
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple, Optional, cast

import torch
from torch.utils._pytree import tree_map
from torchgen.model import (  # pyre-ignore[21]: Undefined import
    FunctionSchema,
    SchemaKind,
)

import spmd.tensor.api as dtensor
from spmd.tensor.placement_types import DTensorSpec, OutputSpecType
from spmd.tensor.utils import (
    unwrap_local_tensor,
    wrap,
    unwrap_schema,
    pack_args_kwargs_with_local_tensor,
)


"""
If _ENABLE_FALLBACK set to False, dispatch will fail when an op doesn't
have a sharding rule registered.
"""
_ENABLE_FALLBACK = False


"""
Print information on ops input shape and sharding for debugging purposes.
"""
_DEBUG_VERBOSE = False


@dataclass
class OpSchema(object):
    """
    OpSchema is a data class that describes an operator input schemas, it
    includes DTensor DTensorSpecs and non-tensor args/kwargs (positional order
    preserved). It is mainly used by the dispatching logic below to run things like
    sharding propagation.

    Sharding propagation rules registered could utilize this data class and
    do inplace update some fields (when necessary, i.e shape related ops) to make
    sure the args/kwargs are legit before passing to the local tensor operator.
    This is the main reason that we don't freeze this dataclass.

    NOTE: greater access to the operator inputs comes with greater responsibility.
    Here are some basic rules about what can be used and what can be changed.

    Args:
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec

    What can be used:
        - every attribute within this class could be read to conduct
          sharding propagation.
    What can be changed:
        - every non-tensor args could be changed to accomodate for local tensor
          operations (i.e. for ops like view/reshape/...)
        - every "DTensorSpec" attribute inside `args_schema`, `kwargs_schema` and
          `args_spec` SHOULD NOT be updated! DTensorSpec are read only and sharding
          propagation shouldn't inplace update them, otherwise the input DTensor
          placements will get implicitly changed and it's error-prone.
    """

    args_schema: Tuple[object, ...]
    kwargs_schema: Dict[str, object]

    @property
    def args_spec(self) -> Tuple[DTensorSpec, ...]:
        """
        args_spec: Tuple[DTensorSpec, ...]: contains a clean list of args spec list
            with NO non-DTensor positional arguments (i.e. int/float/tuple, etc)
            mainly used by sharding propagation to propagate the output spec
        """
        # filter out non-relavant values from args schema to get a clean spec list
        # this would mainly be used by sharding propagation rules
        return tuple(
            item for item in self.args_schema if isinstance(item, DTensorSpec)
        )


@dataclass
class OutputSharding:
    """
    OutputSharding is a data class that is used by the sharding propagation
    rules, it could set the output_spec upon successful propagation, and if
    it failed, output_spec would become None and sharding propagation rules
    could give a list of suggestions for inputs to reshard.

    NOTE: the schema_suggestion generated by sharding propagation should be
    exactly the same as the operator OpSchema, except the DTensor DTensorSpecs
    """

    output_spec: OutputSpecType
    schema_suggestions: Optional[List[OpSchema]] = None
    failed_reason: Optional[str] = None


def _reshape_alias(
    x: torch.Tensor, shape: Tuple[int, ...], strides: Tuple[int, ...]
) -> torch.Tensor:
    return torch.ops.aten.view(x, shape)


_CURRENT_DECOMPOSITION_TABLE: Dict[
    Callable[..., object], Callable[..., object]
] = {torch.ops.aten._reshape_alias.default: _reshape_alias}


def operator_dispatch(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
    op_to_rules: Dict[str, Callable[[OpSchema], OutputSharding]],
    custom_dispatch_ops: Dict[str, Callable[..., object]],
) -> object:
    # first we need to lift some private aten aliases to public calls
    if op_call in _CURRENT_DECOMPOSITION_TABLE:
        return _CURRENT_DECOMPOSITION_TABLE[op_call](*args, **kwargs)

    func_schema = FunctionSchema.parse(str(op_call._schema))
    schema_kind = func_schema.kind()

    # unwrap the args/kwargs schema
    args_schema = tree_map(unwrap_schema, args)
    kwargs_schema = tree_map(unwrap_schema, kwargs)

    op_schema = OpSchema(args_schema, kwargs_schema)

    if _DEBUG_VERBOSE and torch.distributed.get_rank() == 0:
        print(f"{op_call}({op_schema})")
        local_shapes = tree_map(
            lambda t: t.to_local().shape
            if isinstance(t, dtensor.DTensor)
            else None,
            args,
        )
        print(f"    local shapes: {local_shapes}")

    op_key = str(op_call)
    # STEP 0. See if threre're user defined custom aten operator
    # implementations. Custom operators take the highest priority
    if op_key in custom_dispatch_ops:
        # dispatch to user defined custom distributed tensor ops
        return custom_dispatch_ops[op_key](*args, **kwargs)

    sharding_prop_func = op_to_rules.get(op_key, None)

    # step 1. there's sharding propagation rule, run
    # sharding propagation to get output sharding
    if sharding_prop_func is not None:
        output_sharding = sharding_prop_func(op_schema)

        # step 2. if can't get output_spec from sharding
        # propagation (i.e. no rules apply for input
        # placements), we do auto redistribute on inputs
        # to get an eligble input, which we will pick a
        # target schema base on the redistribute cost
        # TODO: implement full auto distribute with a
        # simple cost estimation model
        if output_sharding.output_spec is None:
            # do auto distributed/boxing here
            if output_sharding.schema_suggestions is not None:
                # pick the first suggestion for now,
                target_schema = output_sharding.schema_suggestions[0]
                # run sharding propagation again with target schema
                output_sharding = sharding_prop_func(target_schema)

                local_tensor_args = pack_args_kwargs_with_local_tensor(
                    args,
                    target_schema.args_schema,
                    redistribute_with_schema=True,
                )
                local_tensor_kwargs = pack_args_kwargs_with_local_tensor(
                    kwargs,
                    target_schema.kwargs_schema,
                    redistribute_with_schema=True,
                )

            else:
                raise RuntimeError(
                    f"Sharding propagation failed on op {op_key}!"
                    f"Input schema: {op_schema}."
                    f"Failed reason: {output_sharding.failed_reason}"
                )
        else:
            local_tensor_args = pack_args_kwargs_with_local_tensor(
                args, op_schema.args_schema
            )
            local_tensor_kwargs = pack_args_kwargs_with_local_tensor(
                kwargs, op_schema.kwargs_schema
            )

        # run local op computation with potentially modified args/kwargs
        local_tensor_args = cast(Tuple[object, ...], local_tensor_args)
        local_tensor_kwargs = cast(Dict[str, object], local_tensor_kwargs)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        if schema_kind == SchemaKind.inplace:
            # inplace op should return self instead of re-wrapping
            self = cast(dtensor.DTensor, args[0])
            self._spec = cast(DTensorSpec, output_sharding.output_spec)
            return self
        elif schema_kind == SchemaKind.out:
            # out variant could possibly have multiple out args (i.e. lu_unpack.out)
            output_specs = (
                (output_sharding.output_spec,)
                if not isinstance(output_sharding.output_spec, tuple)
                else output_sharding.output_spec
            )
            out_dts = []
            for i, out in enumerate(func_schema.arguments.out):
                out_dt = cast(dtensor.DTensor, kwargs[out.name])
                out_dt._spec = cast(DTensorSpec, output_specs[i])
                out_dts.append(out_dt)
            return tuple(out_dts) if len(out_dts) > 1 else out_dts[0]
        else:
            return wrap(local_results, output_sharding.output_spec)

    else:
        # step 3. If there's not even one sharding rule
        # implemented for the operator, we fall back to
        # local tensor compute, this is wront currently
        # we will change the behavior to reshard to full
        # replicate and do the computatation
        if not _ENABLE_FALLBACK:
            raise NotImplementedError(
                f"Operator {op_key} does not have a DistributedTensor rule registered."
            )
        # default to local tensor ops, this is wrong
        # but we use it now to enable more tensor point-wise ops
        # TODO: delete this and use replicate (all_gather) as
        # the default fallback.
        else:
            tensor_args = tree_map(unwrap_local_tensor, args)
            tensor_kwargs = tree_map(unwrap_local_tensor, kwargs)
            local_results = op_call(*tensor_args, **tensor_kwargs)
            return wrap(local_results, op_schema.args_spec[0])
