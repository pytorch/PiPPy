from enum import auto, Enum


class BucketingStrategy(Enum):
    FIXED = auto()
    VARIABLE = auto()
    CONSTANT = auto()
