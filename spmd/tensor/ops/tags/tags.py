from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

@dataclass
class DimSet:
    pass

@dataclass
class AllDims(DimSet):
    pass

# keeps dimension
SAME = -1

@dataclass
class AllDBut(DimSet):
    """
    Op operates on a fixed dimension set.

    This is handy for possibly-batched ops.
    e.g. cholesky takes a 2D input but could be batched (3D) so we do fixed=(-1, -2)
    """
    fixed: Optional[Union[int, Tuple]] = None

    """
    Op operates on the dimension given by its input kwarg. This is the name of the kwarg.
    It is usually 'dim' or 'dims'
    """
    kw: Optional[str] = None

    """
    When the input keyword above is not provided, different ops have different
    default behavior to pick the operating dimension. Possible values:

    default=<int>   : the default operating dimension (usually 0)
    default=FLATTEN : means we'll flatten the tensor before operating on its dim 0.
                      and then returns a flattened input array
    default=ALL     : means by default all dimensions are going to be operated on
    """
    default: Union[int, Callable] = None

    """
    Sometimes we know that the input operating dimension is going to generate
    a new dimension in the output, and sometimes we know its size, so we provide it here.
    """
    newsize: Optional[int] = None


@dataclass
class NoDim(DimSet):
    pass

@dataclass
class Op:
    elwise: DimSet = NoDim()
    has_keep_dims: bool = False  # whether the op has a 'keep_dims' option
    linear: bool = False  # whether the op is linear (TODO: could be linear per-input)
    notes: List[str] = None  # custom tags
    move_dims: Union[Callable, List[Tuple]] = None  # whether we're moving input dimensions into the output
    transpose_dims: Union[Callable, List[Tuple]] = None  # whether we are transposing dimensions
    inputwise: bool = False   # whether the op operates on each of its input independently, in which case
                              # lambdas will be called per-input
    # if set , map from input dimension number to output dimension number
    # -1 means that the dimension is new
    dim_map: Optional[List[int]] = None

    shape_argnum: Optional[Union[int, str]] = None

    def __post_init__(self):
        # inputwise means no broadcasting
        if self.elwise == AllDims() and self.inputwise:
            self.dim_map = lambda x: tuple(range(len(x.shape)))

    identity: bool = False,
    preproc: Any = False,  # if you want to run an op before this one for preprocessing the inputs

    alias: Optional[Callable] = None,  # implementation alias, decomposition


# notes
SHARDS_WELL = 'shards_well'

# algebras
LINALG = 'linalg'
BITWISE = 'bitwise'
ADD = '+'  # algebra addition
MUL = '*'  # algebra multiplication
OTHER = '/'

ELWISE = Op(elwise=AllDims())
ALL = 'all'

# output is new singleton dimension
SINGLETON = ('singleton', )

def BROADCAST(input_dim, dim_size):
    """
    output is the broadcast of a singleton input dimension
    """
    return ('broadcast', input_dim, dim_size)

def REDUCE(input_dim, op):
    """
    output is a singleton corresponding to the reduction of an input dimension
    """
    return ('reduce', input_dim, op)

def NEWDIM(val):
    """
    create a new 'generic' dimension with a given size 'val'
    """
    if val == 1:
        # a NEWDIM of size 1 is a singleton!
        return SINGLETON
    else:
        return ('newdim', val)

def MAPDIM(input_dim, new_size, map_func):
    """
    Create an output dimension that is a 'generic' map of a specific input dimension
    # new_size=-1 means keep same dimension size but a dim-wise operation happened
    # new_size=None means "TODO"
    # new_size=-2 means "VARIABLE"
    """
    return ('mapdim', input_dim, new_size, map_func)

def REPEAT(input_dim, times):
    """
    The output dimension is a repeat of the input dimension.
    Repeat happens externally. E.g. for repeat=2 [0, 1, 2] -> [0, 1, 2, 0, 1, 2]
    # Note: this is a specialization of MAPDIM
    """
    if times == 1:
        return input_dim
    elif input_dim == SINGLETON:
        # repeating a singleton is the same as broadcasting it
        return BROADCAST(input_dim, times)
    else:
        return ('repeat', input_dim, times)

def FLATTEN(input_dims: Tuple):
    """
    create a dimension that is a flattened version of the given input dimensions.
    """
    if len(input_dims) == 0:
        # flattening a scalar leads to a singleton
        return SINGLETON
    elif len(input_dims) == 1:
        # flattening a single dimension is no-op
        return input_dims[0]
    else:
        return ('flatten', input_dims)


def COMP(input_dim, group_shape, idx):
    """
    This dimension is a member of a decomposition of the input dim.
    Note , inpnut dim itself could be a FLATTENED input dim.
    E.g. 
    view([6], [2,3]) -> [COMP(0, (2,3,4), 0), COMP(0, (2,3,4), 1), COMP(0, (2,3,4), 2)]
    """
    assert len(group_shape) > 0
    if len(group_shape) == 1:
        # not really a group, just return the input dim back
        assert idx == 0
        return input_dim
    elif group_shape[idx] == 1:
        return SINGLETON
    else:
        # remove singletons from group
        # group_mapping = [(new_index, (shape, old_index)) ...]
        group_mapping = list(enumerate(
            (s, i) for i, s in enumerate(group_shape) if s != 1))
        new_group_shape = tuple(m[1][0] for m in group_mapping)
        new_idx = next(filter(lambda x: x[1][1] == idx, group_mapping))[0]
        ###

        return ('comp', input_dim, new_group_shape, new_idx)


def INPUT_DIM(input_id: Union[int, str], dim_idx):
    """
    specify an input dimension for a multi-input op
    input_id is the args index, or kwarg key for the op input
    """
    return ('input_dim', input_id, dim_idx)


def TENSOR(dims: Tuple, excluded: Tuple = ()):
    """
    Specify a tensor as function of input dimensions.

    Also keep track of input dimensions that are not present in the output,
    due to squeezing or reduction
    """
    return ('tensor', dims, excluded)

