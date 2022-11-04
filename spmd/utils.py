import os
import logging

from . import config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "spmd_format": {"format": "%(name)s: [%(levelname)s] %(message)s"},
    },
    "handlers": {
        "spmd_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "spmd_format",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "spmd": {
            "level": "DEBUG",
            "handlers": ["spmd_console"],
            "propagate": False,
        },
        # TODO(anj): Add loggers for MPMD
    },
    "disable_existing_loggers": False,
}


def init_logging():
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.config.dictConfig(LOGGING_CONFIG)
        spmd_logger = logging.getLogger("spmd")
        spmd_logger.setLevel(config.log_level)
        if config.log_file_name is not None:
            log_file = logging.FileHandler(config.log_file_name)
            log_file.setLevel(config.log_level)
            spmd_logger.addHandler(log_file)
