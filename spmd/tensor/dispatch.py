from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple, Optional, Union

from spmd.tensor.placement_types import PlacementSpec

"""
If set to true, __DEBUG_STRICT will fail when an op doesn't have a sharding rule registered.
"""
_DEBUG_STRICT = False


# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type sould
# be the same set of possiblities.
OutputSpecType = Optional[Union[PlacementSpec, Tuple[PlacementSpec, ...], List[PlacementSpec]]]

@dataclass
class OpSchema(object):
    """
    OpSchema is a data class that is used by the dispatching logic below to run
    things like sharding propagation, it encodes information from unwrapping
    the operator inputs to get the placement specs, and preserve the positional
    information (i.e. the order of args/kwargs) of the schema.

    Sharding propagation rules registered could utilize this data class and
    do inplace update some fields (when necessary, i.e shape related ops) to make
    sure the args/kwargs are legit before passing to the local tensor operator.

    NOTE: greater access to the operator inputs comes with greater responsibility.
    Here are some basic rules about what can be used and what can be changed.

    What can be used:
        - every attribute of this class could be read to conduct sharding propagation.
    What can be changed:
        - every non-tensor args could be changed to accomodate for local tensor
          operations (i.e. for ops like view/reshape/...)
        - outputs_spec should be updated after the sharding propagation rule, if it's
          not updated, we treat the sharding propagation as failed, and will run the
          fallback logic
        - Note that EVERY "Tensor" within args and kwargs must keep UNTOUCHED, you
          should NOT inplace update any tensors inside `args_with_local_tensor` and
          `kwargs_with_local_tensor`. it gonna cause a lot trouble if you do so.
    """
    args_with_local_tensor: Tuple[object, ...]
    kwargs_with_local_tensor: Dict[str, object]
    args_spec: Tuple[PlacementSpec, ...]
    kwargs_spec: Dict[str, PlacementSpec]

    # output specs need to be set as a result of sharding propagation rules
    # if it still be None after the sharding propagation rule, we will treat
    # the sharding propagation as failed.
    outputs_spec: OutputSpecType = None

@dataclass
class OpInfo(object):
    """
    OpInfo is a data class that carries both operator call object and a op schema,
    where the `schema` shall be used by the sharding propagation rules to propagate
    the shardings.
    """
    op_call: Callable[..., object]
    schema: OpSchema


def dispatch_operator(
    op_info: OpInfo,
    op_to_rules: Dict[str, Callable[[OpSchema], OutputSpecType]],
) -> Tuple[object, OutputSpecType]:

    op_key = str(op_info.op_call)
    sharding_prop_func = op_to_rules.get(op_key, None)
    if sharding_prop_func is not None:
        # step 1. there's sharding propagation rule, run
        # sharding propagation to get output placements
        output_placements = sharding_prop_func(op_info.schema)

        # step 2. if can't get output placement (i.e.
        # no ruls apply for input placements), we do
        # auto redistribute on inputs to get an
        # eligble input, which we will pick a target
        # placement base on the redistribute cost
        # TODO: implement auto distribute with simple
        # cost estimation
        if output_placements is None:
            # do auto distributed/boxing here
            raise NotImplementedError("auto redistribute not implemented yet!")

        # run local op computation with potentially modified args/kwargs
        local_results = op_info.op_call(
            *op_info.schema.args_with_local_tensor, **op_info.schema.kwargs_with_local_tensor
        )

        return (local_results, output_placements)
    else:
        # step 3. If there's not even one sharding rule
        # implemented for the operator, we fall back to
        # local tensor compute, this is wront currently
        # we will change the behavior to reshard to full
        # replicate and do the computatation
        if _DEBUG_STRICT:
            raise RuntimeError(
                f"Operator {op_key} does not have a DistributedTensor rule registered."
            )
        # default to local tensor ops, this is wrong
        # but we use it now to enable more tensor point-wise ops
        # TODO: delete this and use replicate (all_gather) as
        # the default fallback.
        else:
            local_results = op_info.op_call(
                *op_info.schema.args_with_local_tensor,
                **op_info.schema.kwargs_with_local_tensor,
            )
            return (local_results, op_info.schema.args_spec[0])
