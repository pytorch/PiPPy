import logging
from typing import Any

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
