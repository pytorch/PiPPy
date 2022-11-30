from enum import auto, Enum

# TODO(chienchin): These are added to make the SPMD flow work. The actual usage
# will be in the following PRs.


class BucketingStrategy(Enum):
    FIXED = auto()
    VARIABLE = auto()
    CONSTANT = auto()
