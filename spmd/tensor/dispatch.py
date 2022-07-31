from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple, Dict

import torch
from spmd.tensor.placement_types import PlacementSpec, Replicate
from spmd.tensor.utils import (
    wrap,
)

"""
If set to true, __DEBUG_STRICT will fail when an op doesn't have a sharding rule registered.
"""
_DEBUG_STRICT = False



@dataclass
class OpInfo(object):
    op_call: Callable
    args_with_local_tensor: Any
    kwargs_with_local_tensor: Any
    args_spec: Any
    kwargs_spec: Any


def dispatch_operator(
    op_info: OpInfo,
    op_to_rules: Dict[str, Callable],
) -> Any:

    op_key = str(op_info.op_call)
    sharding_prop_func = op_to_rules.get(op_key, None)
    if sharding_prop_func is not None:
        # step 1. there's sharding propagation rule, run 
        # sharding propagation to get output placements
        output_placements = sharding_prop_func(op_info)

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

        # run local op computation
        local_results = op_info.op_call(*op_info.args_with_local_tensor, **op_info.kwargs_with_local_tensor)
        print(f">>>> type: {type(local_results)}")

        # rewrap results back to dist tensor and return
        return wrap(local_results, output_placements)
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
        # # default to local tensor ops, this is wrong
        # # but we use it now to enable more tensor op access
        else:
            local_results = op_info.op_call(*op_info.args_with_local_tensor, **op_info.kwargs_with_local_tensor)
            rs = wrap(local_results, op_info.args_spec[0])
            return rs

