# Copyright (c) Meta Platforms, Inc. and affiliates
import shutil
import tempfile
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

import torch.distributed as dist


# wrapper to initialize temp directory for checkpoint
def with_temp_dir(
    func: Optional[  # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        Callable
    ] = None,
) -> Optional[  # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    Callable
]:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore
    ) -> None:
        # Only create temp_dir when rank is 0
        if dist.get_rank() == 0:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = ""
        object_list = [temp_dir]
        # Broadcast temp_dir to all the other ranks
        dist.broadcast_object_list(object_list)
        self.temp_dir = object_list[0]
        print(f"Using temp directory: {self.temp_dir }")
        try:
            func(self)  # type: ignore
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    return wrapper
