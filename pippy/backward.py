# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List

import torch
import pippy.fx


def stage_backward(stage_output, output_grads, input_values, stage_info: str, outputs_with_grads_idxs: List[int]):
    """
    Given the input value(s) and the corresponding gradient for those/that input
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """

    def friendly_debug_info(v):
        if isinstance(v, torch.Tensor):
            return f'Tensor(size={v.shape})'
        else:
            return str(v)

    try:
        stage_output_with_grads = [stage_output[i] for i in outputs_with_grads_idxs]
        output_grads_with_grads = [output_grads[i] for i in outputs_with_grads_idxs]

        # stage_output may be a composite datatype like dict. Extract all individual
        # tensor values here
        stage_output_tensors = []
        output_grad_tensors = []

        def extract_tensors_with_grads(output_val, grad_val):
            if isinstance(output_val, torch.Tensor):
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                assert isinstance(grad_val, (torch.Tensor, type(None))), \
                    f'Expected Tensor or None gradient but got {type(grad_val)}'
                stage_output_tensors.append(output_val)
                output_grad_tensors.append(grad_val)
            elif isinstance(output_val, (tuple, list)):
                if grad_val is None:
                    return
                assert isinstance(grad_val, (tuple, list)), \
                    f'grad_value expected to have type {type(output_val)} but got {type(grad_val)}'
                assert len(output_val) == len(grad_val)
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(ov, gv)
            elif isinstance(output_val, dict):
                if grad_val is None:
                    return
                assert isinstance(grad_val, dict)
                assert set(output_val.keys()) == set(grad_val.keys())
                for k in output_val.keys():
                    extract_tensors_with_grads(output_val[k], grad_val[k])
            else:
                # Output is a non-tensor type; just ignore it
                pass

        extract_tensors_with_grads(stage_output_with_grads, output_grads_with_grads)

        torch.autograd.backward(stage_output_tensors, grad_tensors=output_grad_tensors)

        grad_inputs = []
        for val in input_values:
            if isinstance(val, torch.Tensor):
                grad_inputs.append(val.grad)
            else:
                grad_inputs.append(None)

    except Exception as e:
        exc_msg = f"""
        Failed to run backward stage {stage_info}
        Stage output value: {pippy.fx.node.map_aggregate(stage_output, friendly_debug_info)}
        Output gradient values: {pippy.fx.node.map_aggregate(output_grads, friendly_debug_info)}
        Input values: {pippy.fx.node.map_aggregate(input_values, friendly_debug_info)}
        """
        raise RuntimeError(exc_msg) from e

    barrier_token = None
    return grad_inputs, barrier_token


def sync_barrier(loss, barrier_tokens):
    return loss


# TODO: handling requires_grad=False dynamically. Can we analyze this during initial
# IR emission?
def _null_coalesce_accumulate(lhs, rhs):
    if lhs is None:
        return rhs
    elif rhs is None:
        return lhs
    else:
        return torch.add(lhs, rhs)
