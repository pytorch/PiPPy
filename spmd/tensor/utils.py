# from .api import Tensor

def all_equal(xs):
  xs = list(xs)
  if not xs:
    return True
  return xs[1:] == xs[:-1]
