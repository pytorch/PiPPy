import logging
import logging.config
import os

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "tau_format": {"format": "%(name)s: [%(levelname)s] %(message)s"},
    },
    "handlers": {
        "tau_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "tau_format",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "dtensor": {
            "level": "DEBUG",
            "handlers": ["tau_console"],
            "propagate": False,
        },
        "spmd": {
            "level": "DEBUG",
            "handlers": ["tau_console"],
            "propagate": False,
        },
    },
    "disable_existing_loggers": False,
}


def init_logging(log_level):
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.config.dictConfig(LOGGING_CONFIG)
        dt_logger = logging.getLogger("dtensor")
        dt_logger.setLevel(log_level)
        spmd_logger = logging.getLogger("spmd")
        spmd_logger.setLevel(log_level)
        # TODO(anj): Add option to pipe this to a file


def print0(debug_str):
    # TODO(anj): Add check for torch.distributed initialization.
    if torch.distributed.get_rank() == 0:
        print(debug_str)