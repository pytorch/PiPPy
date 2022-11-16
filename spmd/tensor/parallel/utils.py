import functools

import torch
from spmd.tensor import DeviceMesh, DTensor
from typing import Callable, Optional, Union


def _prepare_output_validate(
    _prepare_output_func: Callable[
        [DTensor, Optional[DeviceMesh], Optional[int]],
        Union[torch.Tensor, DTensor],
    ]
) -> Callable[
    [DTensor, Optional[DeviceMesh], Optional[int]], Union[torch.Tensor, DTensor]
]:
    """
    Inject common validation logics for _prepare_output funcs via this 
    decorator, including verifying that output needs to be a DTensor 
    and only 1D Device Mesh is passed in.
    Example::
        >>> @_prepare_output_validate
        >>> def make_output_shard_1d(args, kwargs):
        >>>   ...
        >>>
        >>> dt = distribute(tensor, device_mesh, [Shard(0)])
        >>> make_output_shard_1d(dt, device_mesh, 1)
        >>> # This will call '_prepare_output_validate' first
    Args:
        _prepare_output_func (Callable): The func we want to inject the 
            validation into.
    Return:
        func (Callable): Same input func with validation logic added.
    """

    @functools.wraps(_prepare_output_func)
    def wrapper(*args, **kwargs):
        assert len(args) >= 2, "_prepare_output need at least two args."
        output = args[0]
        device_mesh = args[1]
        assert isinstance(
            output, DTensor
        ), f"output of Tensor Parallel is actually {type(output)} not DTensor."
        if device_mesh is None:
            device_mesh = output.device_mesh

        assert (
            device_mesh.ndim == 1
        ), f"{_prepare_output_func.__name__}: device mesh is not 1D"
        return _prepare_output_func(*args, **kwargs)

    return wrapper
