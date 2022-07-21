# Copyright (c) Meta Platforms, Inc. and affiliates
# from .api import Tensor


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def all_equal(xs):
    xs = list(xs)
    if not xs:
        return True
    return xs[1:] == xs[:-1]
