import logging
import sys
from types import ModuleType

# log level (levels print what it says + all levels listed below it)
# DEBUG print full traces <-- lowest level + print tracing of every instruction
# INFO print compiler functions + distributed graphs
# WARN print warnings
# ERROR print exceptions
log_level = logging.DEBUG
# Verbose will print full stack traces on warnings and errors
verbose = False

# the name of a file to write the logs to
log_file_name = None


class _AccessLimitingConfig(ModuleType):
    def __setattr__(self, name, value):
        if name not in _allowed_config_names:
            raise AttributeError(f"{__name__}.{name} does not exist")
        return object.__setattr__(self, name, value)


_allowed_config_names = {*globals().keys()}
sys.modules[__name__].__class__ = _AccessLimitingConfig
