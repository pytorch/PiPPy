import logging
from typing import Any

import torch
import torch.distributed as dist


def rank0_debug(logger: logging.Logger, *args: Any, **kwargs: Any) -> None:
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.debug(*args, **kwargs)


def rank0_info(logger: logging.Logger, *args: Any, **kwargs: Any) -> None:
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(*args, **kwargs)


def rank0_warning(logger: logging.Logger, *args: Any, **kwargs: Any) -> None:
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.warning(*args, **kwargs)


def get_mem_usage():
    if dist.is_initialized() and dist.get_rank() == 0:
        return f"allocated/max_allocated {torch.cuda.memory_allocated() / 1024**2}/{torch.cuda.max_memory_allocated() / 1024**2 } MB, \
            reserved/max_reserved {torch.cuda.memory_reserved() / 1024**2 }/{torch.cuda.max_memory_reserved() / 1024**2 } MB"
