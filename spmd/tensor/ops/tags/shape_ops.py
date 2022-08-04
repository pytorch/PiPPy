pymax = max
pyrange = range

from re import L
from sympy import O
import torch
from torch import *
from tags import *


def assert_throw(fn):
    try:
        fn()
    except:
        return
    assert False, 'didnt throw'


def assert_eq(x, v):
    print('')
    if x == v:
        print(x)
    else:
        print(f'error: got {x} vs {v}')
    assert(x == v)
    return x

def assert_close(x, v):
    assert(torch.allclose(x, v))
    return x



def dim_pad_left(input_shape, min_dims):
    return (SINGLETON, ) * pymax(0, min_dims - len(input_shape)) + tuple(pyrange(len(input_shape)))

def dim_pad_right(input_shape, min_dims):
    return  tuple(pyrange(len(input_shape))) + (SINGLETON, ) * pymax(0, min_dims - len(input_shape))


def mapped_broadcast_shapes(shapes):
    # get the largest dim
    max_dim_len = pymax(len(s) for s in shapes)
    # add singleton left padding
    padded_shapes = [
        (1, ) * pymax(0, max_dim_len - len(shape)) + tuple(shape)
        for shape in shapes
    ]
    expanded_shape = [1, ] * max_dim_len
    for i in pyrange(max_dim_len):
        for padded_shape in padded_shapes:
            assert 1 in (expanded_shape[i], padded_shape[i]) or expanded_shape[i] == padded_shape[i], f'{expanded_shape[i]} vs {padded_shape[i]}'
            if padded_shape[i] > expanded_shape[i]:
                expanded_shape[i] = padded_shape[i]
    return tuple(expanded_shape)


def expand(input_shape, shape):
    """ Implements broadcast on multiple dimensions """
    assert len(shape) >= len(input_shape)

    # 1. create padded input dimensions
    padded_input = dim_pad_left(input_shape, len(shape))
    # 2. check that input shapes are compatible
    mapping = []
    for p, desired_s in zip(padded_input, shape):
        if p == SINGLETON:
            actual_s = 1
            assert(desired_s >= 0)
        else:
            actual_s = input_shape[p]
            assert(actual_s == 1 or desired_s == -1 or desired_s == actual_s)
        mapping.append(p if desired_s in (1, -1) or desired_s == actual_s else BROADCAST(p, desired_s))
    return tuple(mapping)


def dimmap_broadcast_tensors(shapes):
    shapes = tuple(shapes)
    out_shape = mapped_broadcast_shapes(tuple(shapes))
    return list(expand(shape, out_shape) for shape in shapes)


def dim_atleast_3d(num_dims):
    if num_dims == 0:
        return (SINGLETON, SINGLETON, SINGLETON)
    elif num_dims == 1:
        return (SINGLETON, 0, SINGLETON)
    elif num_dims == 2:
        return (0, 1, SINGLETON)
    else:
        return tuple(pyrange(num_dims))

def normalize_sizes(*sizes):
    if isinstance(sizes[0], int):
        return sizes
    elif len(sizes) == 1:
        return sizes[0]
    else:
        assert False, 'Size must be int... or tuple'


def normalize_dims(dims, num_dims):
    """
    Takes negative dim numbers into positive dim numbers.
    """
    if dims is None:
        return None
    if isinstance(dims, int):
        return dims if dims >=0 else dims + num_dims
    else:
        return tuple(dim if dim >= 0 else dim + num_dims for dim in dims)
    

import functools
import operator

prod = lambda xs: functools.reduce(operator.mul, xs, 1)


def dim_flatten(shape):
    if len(shape) == 0:
        return (SINGLETON, )
    elif len(shape) == 1:
        return (0, )
    else:
        return (FLATTEN(tuple(pyrange(len(shape)))), )


def dim_movedim(shape_len, input, destination):
    if not isinstance(input, tuple):
        input = (input, )
    if not isinstance(destination, tuple):
        destination = (destination, )
    
    assert len(input) == len(destination)
    input_set = set(input)
    assert len(input_set) == len(input), 'Found repeated input dims'
    assert len(set(destination)) == len(destination), 'Found repeated output dims'
    assert pymax(input) < shape_len
    assert pymax(destination) < shape_len

    dest = [-1, ] * shape_len
    for i, d in zip(input, destination):
        dest[d] = i

    unused_inputs_iter = iter(i for i in pyrange(shape_len) if i not in input_set)
    for i in pyrange(shape_len):
        if dest[i] == -1:
            dest[i] = next(unused_inputs_iter)

    return tuple(dest)

# TODO
def dim_repeat(shape, sizes):
    assert len(sizes) >= len(shape), 'Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor'
    pad = len(sizes) - len(shape)
    return (
        tuple(REPEAT(SINGLETON, s) for s in sizes[:pad]) +
        tuple(REPEAT(i, s) for i, s in enumerate(sizes[pad:]))
    )

def infer_size(total_size, sizes):
    """
    One dimension input to view may be "-1".
    Infer the size of this dimension given the total_size.
    """
    infers = [i for i, s in enumerate(sizes) if s == -1]
    size = prod(sizes)
    assert len(infers) <= 1, 'can only infer one size'
    if infers:
        size = -size
        missing_size = total_size // size
        assert total_size % size == 0, f"size inferred for -1 is not integral {sizes} should have {total_size} elements."
        return [s if s != -1 else missing_size for s in sizes]
    assert size == total_size, f"sizes do not match {total_size} vs {size}"
    return sizes


def view_groups(from_size, to_size):
    """
    Split up the total view into smaller groups of dimensions whose size will match:
    view_groups([3, 4, 5], [12, 5]) -> [([3, 4], [12]), ([5], [5])]
    """
    from_nelem = prod(from_size)
    to_size = infer_size(from_nelem, to_size)

    assert from_nelem == prod(to_size), 'Total view shape does not add up'

    # from_it = iter(from_size)
    # to_it = iter(to_size)

    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)

    result = []
    result_dim = []

    #while True:
    while from_idx < from_len or to_idx < to_len:
        # t, f = next(to_it, None), next(from_it, None)


        # if t is None and f is None:
        #     break

        from_group, to_group = [], []
        from_group_dim, to_group_dim = [], []

        # if f is None:
        if from_idx >= from_len:
            f = 1
        else:
            f = from_size[from_idx]
            from_group.append(f)
            from_group_dim.append(from_idx)
            from_idx += 1

        # if t is None:
        if to_idx >= to_len:
            t = 1
        else:
            t = to_size[to_idx]
            to_group.append(t)
            to_group_dim.append(to_idx)
            to_idx += 1

        # if any of the groups is singleton, great, we need to backtrack though
        if f == 1 and t != 1:
            # produces ([1], [])
            to_idx -= 1
            to_group = []
            to_group_dim = []
        elif f != 1 and t == 1:
            # produces ([], [1])
            from_idx -= 1
            from_group = []
            from_group_dim = []
        else:
            # produces ([1], [1]),  ([2], [2]), ([2,3], [6])
            while f != t:
                if f < t:
                    #nf = next(from_it)
                    nf = from_size[from_idx]

                    from_group.append(nf)
                    from_group_dim.append(from_idx)

                    from_idx += 1
                    f *= nf
                else:
                    # nt = next(to_it)
                    nt = to_size[to_idx]

                    to_group.append(nt)
                    to_group_dim.append(to_idx)

                    to_idx += 1
                    t *= nt

        result.append((from_group, to_group))
        result_dim.append((from_group_dim, to_group_dim))

        result_pp = []
        for (_, r), (f, t) in zip(result, result_dim):
            if len(t) == 0:
                # we are removing the dimension
                # TODO: we need to mark it for autograd purposes later
                continue
            # removing singleton dimensions from the interior of the
            # flattened tuple
            # TODO: we need to mark it for autograd purposes later
            ff = FLATTEN(tuple(fi for fi in f if from_size[fi] > 1))
            tr = tuple(r)
            result_pp += [COMP(ff, tr, i) for i in pyrange(len(r))]
        result_pp = tuple(result_pp)

    return result, result_pp


assert_eq(
    view_groups([3, 4, 5], [12, 5]),
    (
        [
            ([3, 4], [12]),
            ([5], [5]),
        ],
        (
            FLATTEN((0, 1)),
            2,
        )
    )
)

assert_eq(
    view_groups([2,3,4,5,7], [12, 70]),
    (
        [
            ([2,3,4,5,7], [12, 70]),
        ],
        (
            COMP(FLATTEN((0,1,2,3,4)), (12, 70), 0),
            COMP(FLATTEN((0,1,2,3,4)), (12, 70), 1),
        )
    )
)

assert_eq(
    view_groups([2,3,4,5,7], [3,8, 7, 5]),
    (
        [
            ([2,3,4], [3,8]),
            ([5,7], [7,5]),
        ],
        (
            COMP(FLATTEN((0,1,2)), (3,8), 0),
            COMP(FLATTEN((0,1,2)), (3,8), 1),
            COMP(FLATTEN((3,4)), (7,5), 0),
            COMP(FLATTEN((3,4)), (7,5), 1),
        )
    )
)

assert_eq(
    view_groups([3,4,8,3], [12, 4, 2, 3]),
    (
        [
            ([3, 4], [12]), 
            ([8], [4, 2]),
            ([3], [3])
        ],
        (
            FLATTEN((0, 1)),
            COMP(2, (4, 2), 0),
            COMP(2, (4, 2), 1),
            3,
        )
    )
)

assert_eq(
    view_groups([3,24], [1, 3,  2,4,1,3, 1]),
    (
        [
            ([], [1]),
            ([3], [3]),
            ([24], [2, 4, 1, 3]),
            ([], [1]),
        ],
        (
            SINGLETON,
            0,
            COMP(1, (2,4,3), 0), # note dimension removed
            COMP(1, (2,4,3), 1),
            SINGLETON,
            COMP(1, (2,4,3), 2),
            SINGLETON,
        )
    )
)

assert_eq(
    view_groups([1,1,3,2,1,1], [6,1,1,1]),
    (
        [
            ([1], []),
            ([1], []),
            ([3,2], [6]),
            ([1], [1]),
            ([1], [1]),
            ([], [1]),
        ],
        (
            FLATTEN((2,3)),
            SINGLETON,
            SINGLETON,
            SINGLETON,
        )
    ),
)
assert_eq(
    view_groups([1,1,12,1,1,1,2,5,1], [3,4,1,10]),
    (
        [
            ([1], []),
            ([1], []),
            ([12], [3,4]),
            ([1], [1]),
            ([1], []),
            ([1], []),
            ([2,5], [10]),
            ([1], []),
        ],
        (
            COMP(2, (3,4), 0),
            COMP(2, (3,4), 1),
            SINGLETON,
            FLATTEN((6,7))
        )
    )
)
assert_eq(
    view_groups([2,3,4], [2, -1, 4]),
    (
        [([2], [2]), ([3], [3]), ([4], [4])],
        (0, 1, 2)
    )
)


# TODO(test)
def first_dim3(input_shape, other_shape):
    assert len(input_shape) == len(other_shape)
    for dim_idx, (i, o) in enumerate(zip(input_shape, other_shape)):
        if i == 3 and o == 3:
            return dim_idx
    assert False, 'Could not find dimension where both are shape 3.'


def dim_squeeze(shape, dim=None):
    dim = normalize_dims(dim, len(shape))
    if dim == None:
        # remove all singletons
        return tuple(i for i, s in enumerate(shape) if s != 1)
    else:
        result = tuple(pyrange(len(shape)))
        # special case zero-d
        if len(shape) == 0 and dim == 0:
            return ()
        if shape[dim] != 1:
            # if the given dimension is not a singleton, just return the original
            return result
        else:
            return result[:dim] + result[dim+1:]


def dim_stack(shapes, dim):
    assert len(set(shapes)) == 1, 'All tensors must have the same shape.'
    shape = tuple(pyrange(len(shapes[0])))
    return shape[:dim] + (NEWDIM(len(shapes)), ) + shape[dim:]
 

def dim_transpose(shape, dim1, dim2):
    assert dim1 < len(shape)
    assert dim2 < len(shape)
    shape = list(pyrange(len(shape)))
    shape[dim1] = dim2
    shape[dim2] = dim1
    return tuple(shape)


def dim_tile(shape, dims):
    if len(dims) < len(shape):
        dims = (1, ) * (len(shape) - len(dims)) + dims
    return dim_repeat(shape, dims)


def dim_unbind(shape, dim):
    dims = tuple(pyrange(len(shape)))
    return dims[:dim] + dims[dim+1:]


def dim_unsqueeze(shape, dim):
    dims = tuple(pyrange(len(shape)))
    return dims[:dim] + (SINGLETON, ) + dims[dim:]


ops = {
    atleast_1d: Op(inputwise=True, dim_map=lambda x: dim_pad_left(x.shape, 1)),
    atleast_2d: Op(inputwise=True, dim_map=lambda x: dim_pad_left(x.shape, 2)),
    atleast_3d: Op(inputwise=True, dim_map=lambda x: dim_atleast_3d(len(x.shape))),
    # broadcast_shapes: None,  # need to figre out whether shapes themselves need to be sharded? probably not.
    broadcast_tensors: Op(dim_map=lambda *tensors: dimmap_broadcast_tensors(t.shape for t in tensors)),
    broadcast_to: Op(dim_map=lambda input, shape: expand(input.shape, shape)),
    cat: Op(elwise=AllDBut(kw='dim', default=0)),
    chunk: Op(elwise=AllDBut(kw='dim', default=0)),  # variant of slice
    column_stack: Op(
        preproc=lambda tensors: tuple(dim_pad_right(t.shape, 2) for t in tensors),
        elwise=AllDBut(fixed=1)),
    'Tensor.contiguous': Op(identity=True), # tensor call
#    'dsplit': None, # not found?
    dstack: Op(
        preproc=lambda tensors: tuple(dim_atleast_3d(len(t.shape)) for t in tensors),
        elwise=AllDBut(fixed=2),
    ),
    'Tensor.expand': Op(dim_map=lambda self, *sizes: expand(self.shape, normalize_sizes(*sizes))),
    'Tensor.expand_as': Op(dim_map=lambda self, other: expand(self.shape, other.shape)),
    flatten: Op(dim_map=lambda tensor: dim_flatten(tensor.shape)),
    flip: Op(elwise=AllDBut(kw='dims', newsize=SAME)),
    fliplr: Op(elwise=AllDBut(fixed=1, newsize=SAME)),
    flipud: Op(elwise=AllDBut(fixed=0, newsize=SAME)),
    'H': Op(dim_map=lambda input: dim_transpose(input.shape, -1, -2)),
    hsplit: Op(elwise=AllDBut(fixed=lambda input, *_: 0 if len(input.shape) < 2 else 1)),
    hstack: Op(elwise=AllDBut(fixed=lambda inputs: 0 if len(inputs[0].shape) < 2 else 1)),
    'mH': Op(dim_map=lambda input: dim_transpose(input.shape, -1, -2)),
    movedim: Op(dim_map=lambda input, source, destination: dim_movedim(len(input.shape), source, destination)),
    'mT': Op(dim_map=lambda input: dim_transpose(input.shape, -1, -2)),
    narrow: Op(elwise=AllDBut(kw='dim')),
    narrow_copy: Op(elwise=AllDBut(kw='dim')),
    permute: Op(dim_map=lambda input, dims: dims),
    ravel: Op(dim_map=lambda tensor: dim_flatten(tensor.shape)),
    'Tensor.repeat': Op(dim_map=lambda self, *sizes: dim_repeat(self.shape, sizes)),
    'repeat_interleave.self_Tensor': Op(elwise=AllDBut(kw='dim', default=FLATTEN)),
    'repeat_interleave.Tensor': Op(elwise=AllDBut(fixed=0)),
    reshape: Op(dim_map=lambda input, shape: view_groups(input.shape, shape)[1]),
    'Tensor.reshape_as': Op(dim_map=lambda self, other: view_groups(self.shape, other.shape)[1]),
    'Tensor.resize_': Op(),  # cant' do much here
    'Tensor.resize_as_': Op(),
    roll: Op(elwise=AllDBut(kw='dims', default=ALL)),
    rot90: Op(elwise=AllDBut(kw='dims')),
    split: Op(elwise=AllDBut(kw='dim', default=0)),
    'Tensor.split_with_sizes': Op(elwise=AllDBut(kw='dim', default=0)),
    squeeze: Op(dim_map=lambda input, dim=None: dim_squeeze(input.shape, dim)),
    stack: Op(dim_map=lambda tensors, dim=0: dim_stack(tuple(t.shape for t in tensors), dim)),
    swapaxes: Op(dim_map=lambda input, axis0, axis1: dim_transpose(input.shape, axis0, axis1)),
    'T': Op(dim_map=lambda input: dim_transpose(input.shape, -1, -2)),
    't': Op(dim_map=lambda input: dim_transpose(input.shape, -1, -2)),
    tensor_split: Op(elwise=AllDBut(kw='dim', default=0)),
    tile: Op(dim_map=lambda input, dims: dim_tile(input.shape, dims)),
    transpose: Op(dim_map=lambda input, dim0, dim1: dim_transpose(input.shape, dim0, dim1)),
    unbind: Op(dim_map=lambda input, dim=0: dim_unbind(input.shape, dim)),  # inverse of stack
#    'unflatten': None,  # couldn't find it.
    'Tensor.unfold': Op(),  # TODO: this seems related to convolution
    unsqueeze: Op(dim_map=lambda input, dim: dim_unsqueeze(input.shape, dim)),
    'Tensor.view': Op(dim_map=lambda input, *shape: view_groups(input.shape, shape)[1]),
    'Tensor.view_as': Op(dim_map=lambda self, other: view_groups(self.shape, other.shape)[1]),
    vsplit: Op(elwise=AllDBut(fixed=0)),
    vstack: Op(elwise=AllDBut(fixed=0)),
}


# test helper
def decorate_dim_map(op):
    op._dim_map_test_calls = []
    op._dim_map_test_calls_did_throw = []
    op._orig_dim_map = op.dim_map

    def _record_dim_map_test_call(*args, **kwargs):
        op._dim_map_test_calls.append((args, kwargs))
        try:
            result = op._orig_dim_map(*args, **kwargs)
            op._dim_map_test_calls_did_throw.append(False)
            return result
        except:
            op._dim_map_test_calls_did_throw.append(True)
            raise

    op.dim_map = _record_dim_map_test_call


for op in ops.values():
    if op.dim_map:
        decorate_dim_map(op)


assert(ops[atleast_1d].dim_map(randn(())) == (SINGLETON, ))
assert(ops[atleast_1d].dim_map(randn(2)) == (0, ))
assert(ops[atleast_1d].dim_map(randn(2, 3)) == (0, 1))

assert(ops[atleast_2d].dim_map(randn(())) == (SINGLETON, SINGLETON))
assert(ops[atleast_2d].dim_map(randn(2)) == (SINGLETON, 0))
assert(ops[atleast_2d].dim_map(randn(2, 3)) == (0, 1))
assert(ops[atleast_2d].dim_map(randn(2, 3, 4)) == (0, 1, 2))

assert(ops[atleast_3d].dim_map(randn(())) == (SINGLETON, SINGLETON, SINGLETON))
assert(ops[atleast_3d].dim_map(randn(2)) == (SINGLETON, 0, SINGLETON))
assert(ops[atleast_3d].dim_map(randn(2, 3)) == (0, 1, SINGLETON))
assert(ops[atleast_3d].dim_map(randn(2, 3, 4)) == (0, 1, 2))
assert(ops[atleast_3d].dim_map(randn(2, 3, 4, 2)) == (0, 1, 2, 3))

assert_throw(lambda: ops[broadcast_to].dim_map(randn(2,3), (1,2,4)))

assert_eq(ops[broadcast_to].dim_map(rand(2,3), (1,2,3)), (SINGLETON, 0, 1))
assert_eq(ops[broadcast_to].dim_map(rand(2,3), (4,2,3)), (BROADCAST(SINGLETON, 4), 0, 1))
assert_eq(ops[broadcast_to].dim_map(rand(2,1,3), (2,2,2,3)), (BROADCAST(SINGLETON, 2), 0, BROADCAST(1, 2), 2))
assert_eq(ops[broadcast_to].dim_map(rand(2,3), (-1,3)), (0, 1))
assert_eq(ops[broadcast_to].dim_map(rand(2,1,3), (-1,1,3)), (0, 1, 2))


assert_eq(mapped_broadcast_shapes(((1,2,4), (2,4), (4, ), (1, 1, 1), (3,1,1))), (3,2,4))

assert_eq(
    ops[broadcast_tensors].dim_map(randn(1,2,4), randn(2,4), randn(4), randn(1, 1, 1)),
    [
        (0, 1, 2),
        (SINGLETON, 0, 1),
        (SINGLETON, BROADCAST(SINGLETON, 2), 0),
        (0, BROADCAST(1, 2), BROADCAST(2, 4)),
    ],
)

assert_eq(
    ops[broadcast_to].dim_map(randn(3,1,2), (2,3,4,2)),
    (BROADCAST(SINGLETON, 2), 0, BROADCAST(1, 4), 2)
)

assert_eq(
    ops[column_stack].preproc((randn(2,3), randn(2), randn(()))),
    ((0, 1), (0, SINGLETON), (SINGLETON, SINGLETON))
)

assert_eq(ops[dstack].preproc((randn(2), rand(1, 2, 5))), ((SINGLETON, 0, SINGLETON), (0, 1, 2)))


assert_eq(
    ops['Tensor.expand'].dim_map(randn(2,1,3,1), 3,2,4,-1,2),
    (BROADCAST(SINGLETON, 3), 0, BROADCAST(1, 4), 2, BROADCAST(3, 2))
)

assert_eq(
    ops['Tensor.expand'].dim_map(randn(2,1,3,1), (3,2,4,-1,2)),
    (BROADCAST(SINGLETON, 3), 0, BROADCAST(1, 4), 2, BROADCAST(3, 2))
)

assert_eq(
    ops['Tensor.expand_as'].dim_map(randn(2,1,3,1), randn(3,2,4,3,2)),
    (BROADCAST(SINGLETON, 3), 0, BROADCAST(1, 4), 2, BROADCAST(3, 2))
)

assert_eq(ops[flatten].dim_map(randn(2,3)), (FLATTEN((0, 1)), ))
assert_eq(ops[flatten].dim_map(randn(4)), (0, ))
assert_eq(ops[flatten].dim_map(randn(())), (SINGLETON, ))

assert_eq(ops[hsplit].elwise.fixed(randn(2, 3), 2), 1)
assert_eq(ops[hsplit].elwise.fixed(randn(2), 2), 0)
assert_eq(ops[hsplit].elwise.fixed(randn(2, 3), 2), 1)

assert_eq(ops[hstack].elwise.fixed((randn(2, 4, 8), randn(2, 6, 8))), 1)
assert_eq(ops[hstack].elwise.fixed((randn(2), randn(3))), 0)

assert_eq(dim_movedim(4, 1, 2), (0, 2, 1, 3))
assert_eq(dim_movedim(3, 1, 0), (1, 0, 2))
assert_eq(dim_movedim(3, (1, 2), (0, 1)), (1, 2, 0))
assert_eq(dim_movedim(3, (0, 2, 1), (2, 1, 0)), (1, 2, 0))
assert_eq(dim_movedim(2, (1, 0), (0, 1)), (1, 0))

assert_eq(ops[movedim].dim_map(randn(3,2,1), (1, 2), (0, 1)), (1, 2, 0))

assert_eq(ops[permute].dim_map(randn(2,3,4), (2, 0, 1)), (2, 0, 1))

assert_eq(ops[ravel].dim_map(randn(2,3)), (FLATTEN((0, 1)), ))
assert_eq(ops[ravel].dim_map(randn(4)), (0, ))
assert_eq(ops[ravel].dim_map(randn(())), (SINGLETON, ))

assert_eq(
    ops['Tensor.repeat'].dim_map(randn(2,3), 1, 2, 1, 1, 2),
    (SINGLETON, BROADCAST(SINGLETON, 2), SINGLETON, 0, REPEAT(1, 2))
)

assert_eq(
    ops[reshape].dim_map(randn(2,4,8), (8, 8)),
    (FLATTEN((0, 1)), 2),
)

assert_eq(
    ops['Tensor.reshape_as'].dim_map(randn(2,4,8), randn(8, 8)),
    (FLATTEN((0, 1)), 2),
)

assert_eq(ops[squeeze].dim_map(randn(2,1,4,1)), (0,2))
assert_eq(ops[squeeze].dim_map(randn(2,1,4,1), dim=0), (0,1,2,3))
assert_eq(ops[squeeze].dim_map(randn(2,1,4,1), dim=1), (0,2,3))
assert_eq(ops[squeeze].dim_map(randn(()), dim=0), ())
assert_eq(ops[squeeze].dim_map(randn(1), dim=0), ())
assert_eq(ops[squeeze].dim_map(randn(2), dim=0), (0, ))


assert_eq(ops[stack].dim_map((randn(2,4), randn(2,4), randn(2,4))), (NEWDIM(3), 0, 1))
assert_eq(ops[stack].dim_map((randn(3), randn(3)), dim=1), (0, NEWDIM(2)))

assert_eq(ops[swapaxes].dim_map(randn(2,5,4,5), 2, 0), (2,1,0,3))

assert_eq(
    ops[tile].dim_map(randn(2,3), (1, 2, 1, 1, 2)),
    (SINGLETON, BROADCAST(SINGLETON, 2), SINGLETON, 0, REPEAT(1, 2))
)
assert_eq(
    ops[tile].dim_map(randn(4, 2,3), (1, 3, )),
    (0, 1, REPEAT(2, 3)),
)

assert_eq(ops[transpose].dim_map(randn(2,5,4,5), 2, 0), (2,1,0,3))

# TODO: figure out multiple outputs
assert_eq(ops[unbind].dim_map(randn(4,2,3)), (1, 2))
assert_eq(ops[unbind].dim_map(randn(4,2,3), dim=1), (0, 2))

assert_eq(ops[unsqueeze].dim_map(randn(4,2,3), 1), (0, SINGLETON, 1, 2))

assert_eq(
    ops['Tensor.view'].dim_map(randn(2,4,8), 8, 8),
    (FLATTEN((0, 1)), 2),
)

assert_eq(
    ops['Tensor.view_as'].dim_map(randn(2,4,8), randn(8, 8)),
    (FLATTEN((0, 1)), 2),
)

assert_eq(
    ops['Tensor.view'].dim_map(randn(1, 1, 4), -1),
    (2, ),
)

assert_eq(
    ops['Tensor.view'].dim_map(randn(1, 1, 4, 2), -1),
    (FLATTEN((2, 3)), ),
)

assert_eq(
    ops['Tensor.view'].dim_map(randn(1, 1, 4, 1, 2, 1), -1),
    (FLATTEN((2, 4)), ),
)




from torch.utils._pytree import tree_flatten

def expected_shape(in_shape, rule):
    out_shape = [0] * len(rule)
    for dim, cmd in enumerate(rule):
        if isinstance(cmd, int):
            out_shape[dim] = in_shape[cmd]
        elif isinstance(cmd[0], tuple) or isinstance(cmd[0], int):
            assert False, 'shouldnt happen'
            # this is actually a tensor list; go a level lower
            out_shape[dim] = expected_shape(in_shape, cmd)
        elif cmd[0] == 'singleton':
            out_shape[dim] = 1
        elif cmd[0] == 'flatten':
            out_shape[dim] = 1
            for i in cmd[1]:
                out_shape[dim] *= in_shape[i]
        elif cmd[0] == 'comp':
            out_shape[dim] = cmd[1][cmd[2]]
        elif cmd[0] == 'broadcast':
            # this is actually wrong for broadcast_tensors
            # if isinstance(cmd[1], int):
            #    assert in_shape[cmd[1]] == 1, f'in_shape: {in_shape}, expected singleton at {cmd[1]}'
            out_shape[dim] = cmd[2]
        elif cmd[0] == 'repeat':
            out_shape[dim] = in_shape[cmd[1]] * cmd[2]
        elif cmd[0] == 'newdim':
            out_shape[dim] = cmd[1]
        else:
            raise RuntimeError(f'cmd not found: {cmd}, in rule: {rule}')
    return out_shape


class _Partial:
    def __repr__(self):
        return '+'


class Replicate:
    def __repr__(self):
        return 'r'


class Shard:
    def __init__(self, dim):
        self.dim = dim

    def __repr__(self):
        return f's({self.dim})'


from collections import defaultdict
from copy import copy


def propagate_sharding(in_shard, in_shape, rule, mesh_sizes):
    assert len(in_shard) == len(mesh_sizes)
    out_shard = copy(in_shard) # will change

    sharded_in_dims = set(s.dim for s in in_shard if isinstance(s, Shard))

    def get_input_dim(cmd):
        if isinstance(cmd, int):
            return cmd if cmd in sharded_in_dims else None
        elif cmd[0] == 'flatten':
            for in_dim in cmd[1][1:]:
                assert not in_dim in sharded_in_dims, 'Only the first member of a FLATTEN group can be sharded'
            return cmd[1][0] if cmd[1][0] in sharded_in_dims else None
        elif cmd[0] == 'comp':
            if cmd[2] > 0:
                # we will shard only on the first dimension of the group
                return None
            else:
                in_dim = get_input_dim(cmd[1])
                if in_dim is None:
                    # in case input dim is not sharded
                    return None
                # we need to check that the input dimension is divisble
                # by the size of the submesh we're sharding it on
                submesh_size = 1
                for size, shard in zip(mesh_sizes, in_shard):
                    if isinstance(shard, Shard) and shard.dim == in_dim:
                        submesh_size *= size
                out_size = cmd[1][0]
                assert out_size % submesh_size, 'Resulting dimension size inot divisible by its mesh dimension.'
                return in_dim
        elif cmd[0] in ('singleton', 'broadcast', 'newdim'):
            # no-op for sharding (those dimensions will be replicated by default)
            return None
        elif cmd[0] == 'repeat':
            assert not cmd[1] in sharded_in_dims, 'Cannot tile sharded dimension.'
            return None
        else:
            raise RuntimeError(f'cmd not found: {cmd}, in rule: {rule}')
 
    dim_map = {}
    for dim, cmd in enumerate(rule):
        in_dim = get_input_dim(cmd)
        if in_dim is not None:
            dim_map[in_dim] = dim

    return [Shard(dim_map[s.dim]) if isinstance(s, Shard) else s for s in in_shard]


for op, spec in ops.items():
    if spec.dim_map:
        print(op)
        if isinstance(op, str):
            splat = op.split('.')
            if len(splat) == 2:
                cls, meth = splat
                assert cls == 'Tensor'
                op = getattr(Tensor, meth)
            else:
                assert len(splat) == 1
                op = getattr(Tensor, splat[0])


        # pick some random mesh size
        mesh_sizes = [2, 4]
        import random

        for (args, kwargs), did_throw in zip(spec._dim_map_test_calls, spec._dim_map_test_calls_did_throw):
            try:
                rules = spec._orig_dim_map(*args, **kwargs)
                outputs = op(*args, **kwargs)
                assert not did_throw
                print('--------')
                print(rules)



                flat_args, _ = tree_flatten(args)
                output_shapes = [a.shape for a in outputs] if isinstance(outputs, tuple) else outputs.shape

                if isinstance(rules, list):
                    assert len(flat_args) == len(rules)
                    expected = [expected_shape(arg.shape, rule) for arg, rule in zip(flat_args, rules)]
                else:
                    in_shape = flat_args[0].shape
                    expected = expected_shape(in_shape, rules)

                    no_shard_dims = set()
                    for rule in rules:
                        if isinstance(rule, tuple):
                            if rule[0] == 'repeat':
                                no_shard_dims.add(rule[1])
                            if rule[0] == 'flatten':
                                no_shard_dims |= set(rule[1][1:])
                            if rule[0] == 'comp':
                                if isinstance(rule[1], tuple) and rule[1][0] == 'flatten':
                                    no_shard_dims |= set(rule[1][1][1:])

                    def pick_sharding(in_shape, mesh_dim_size):
                        # - do not shard on singleton dimensions (this wouldn't happen assuming DT is well constructed)
                        # - only shard on first dim of flattened groups


                        return random.choice([Replicate(), _Partial()] + [
                            Shard(i) for i, s in enumerate(in_shape) if s > 1 and i not in no_shard_dims]
                        )

                    # some random in_shard
                    in_shard = [pick_sharding(in_shape, d) for d in mesh_sizes]
                    print(f'in_shard: {in_shard}')
                    print(f'in_shape: {in_shape}')
                    print(f'rules: {rules}')

                    out_shard = propagate_sharding(in_shard, in_shape, rules, mesh_sizes)
                    print(f'out_shard: {out_shard}')

                print(len(flat_args), [arg.shape if isinstance(arg, Tensor) else None for arg in flat_args], output_shapes, expected)

                
            except Exception as e:
                if not did_throw:
                    print('------- expected call not to throw but got error -------')
                    print(op)
                    print(args)
                    print(kwargs)
                    raise e
        #print(spec._dim_map_test_calls)