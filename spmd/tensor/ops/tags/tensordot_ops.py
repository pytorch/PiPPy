pymax = max
pyrange = range

import torch
from torch import *
from tags import *


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


def dim_einsum(pattern, *a):
    # TODO
    pass

def einsum_impl(pattern, *a, out=None):
    if isinstance(pattern, DimSpec):
        pattern = str(pattern)
    assert out is None, 'In place einsum not supported.'
    return torch.einsum(pattern, *a)


def alias_gemm(pattern, input, m1, m2, beta=1, alpha=1, out=None):
    """
    einsum + add
    """
    if beta == 0 and alpha == 1:
        return einsum_impl(pattern, m1, m2, out=out)
    elif beta == 1 and alpha == 1:
        return torch.add(input, einsum_impl(pattern, m1, m2), out=out)
    else:
        return torch.add(
            beta * input,
            alpha * einsum_impl(pattern, m1, m2),
            out=out)



from pattern import DimSpec


# TODO: test
def chain_mm_pattern(num_mats):
    dims = 'abcdefghijklmnopqrstuvwxyz'
    assert num_mats < len(dims), f'Suppported at max {len(dims)} matrices'
    return ','.join(dims[i] + dims[i+1] for i in pyrange(num_mats)) + f'->a{dims[num_mats]}'

assert_eq(chain_mm_pattern(3), 'ab,bc,cd->ad')
assert_eq(chain_mm_pattern(1), 'ab->ab')
assert_eq(chain_mm_pattern(2), 'ab,bc->ac')


def dim_tensordot(shape_a, shape_b, dims=2):
    if isinstance(dims, int):
        dims = (
            tuple(pyrange(-dims, 0)),
            tuple(pyrange(0, dims)),
        )
    assert isinstance(dims, tuple)
    assert len(dims) == 2
    assert isinstance(dims[0], tuple)
    assert isinstance(dims[1], tuple)
    assert len(dims[0]) == len(dims[1])
    dims = (
        normalize_dims(dims[0], len(shape_a)),
        normalize_dims(dims[1], len(shape_b)),
    )

    all_dims = DimSpec.gen_dims(len(shape_a) + len(shape_b))

    # for first input just pick the first dimensions
    in1_dims = all_dims[:len(shape_a)]
    # for the first input, first pick the reducing dimensions
    in2_dims = [None] * len(shape_b)
    for d1, d2 in zip(dims[0], dims[1]):
        in2_dims[d2] = in1_dims[d1]
    # then complete the second input with non-reducing dimensions
    it = iter(all_dims[len(shape_a):])
    for i in pyrange(len(in2_dims)):
        if in2_dims[i] is None:
            x  = next(it)
            in2_dims[i] = next(it)
    in2_dims = tuple(in2_dims)
    # output contains all dims that appear in one and only one of the inputs
    out_dims = tuple(d for d in all_dims if (d in in1_dims) != (d in in2_dims))
    return DimSpec((in1_dims, in2_dims), (out_dims, ))


def tensordot(a, b, dims=2, out=None):
    pass 



assert_eq(str(dim_tensordot((2,3,4),(3,4,5), 2)), 'abc,bce->ae')


ops = {
    addbmm: Op(alias=lambda input, batch1, batch2, *, beta=1, alpha=1, out=None: alias_gemm(
        'bik,bkj->ij', input, batch1, batch2, beta=beta, alpha=alpha, out=out)),
    addmm: Op(alias=lambda input, mat1, mat2, *, beta=1, alpha=1, out=None: alias_gemm(
        'ik,kj->ij', input, mat1, mat2, beta=beta, alpha=alpha, out=out)),
    addmv: Op(alias=lambda input, mat, vec, *, beta=1, alpha=1, out=None: alias_gemm(
        'ik,k->i', input, mat, vec, beta=beta, alpha=alpha, out=out)),
    addr: Op(alias=lambda input, vec1, vec2, *, beta=1, alpha=1, out=None: alias_gemm(
        'i,j->ij', input, vec1, vec2, beta=beta, alpha=alpha, out=out)),
    baddbmm: Op(alias=lambda input, batch1, batch2, *, beta=1, alpha=1, out=None: alias_gemm(
        'bik,bkj->bij', input, batch1, batch2, beta=beta, alpha=alpha, out=out)),
    block_diag: Op(),  # all dimensions change for this one
    bmm: Op(alias=lambda input, mat2, *, out=None: torch.einsum(
        'bik,bkj->bij', input, mat2, out=out)),
    mm: Op(alias=lambda input, mat2, *, out=None: torch.einsum(
        'ik,kj->ij', input, mat2, out=out)),
    cartesian_prod: Op(elwise=0),  # TODO
    chain_matmul:  Op(alias=lambda *matrices, out=None: torch.einsum(
        chain_mm_pattern(len(matrices)), *matrices, out=out)),
    cholesky: Op(elwise=AllDBut(fixed=(-2, -1))),  # possibly batched 
    cholesky_inverse: Op(elwise=AllDBut(fixed=(-2, -1))),  # possibly batched 
    cholesky_solve: Op(elwise=AllDBut(fixed=(-2, -1))), # for both inputs 
    cross: Op(elwise=AllDBut(kw='dim', default=lambda input, other: first_dim3(input.shape, other.shape))),
    diag: Op(), # makes more sense to replicate and shard
    diag_embed: Op(),
    diagflat: Op(),
    diagonal: Op(),
    dot: Op(alias=lambda input, other, *, out=None: torch.einsum(
        'i,i->', input, other, out=out)),
    eig: Op(),
    einsum: Op(dim_map=lambda equation, *operands: dim_einsum(equation, *operands)),
    frobenius_norm: None,  # could not find spec
    geqrf: Op(),
    ger: Op(alias=lambda input, vec2, *, out=None: einsum_impl('i,j->ij', input, vec2, out=out)),
    inner: Op(alias=lambda input, other, *, out=None: torch.tensordot(input, other, dims=((-1, ), (-1, )))),
    

    outer: Op(alias=lambda input, vec2, *, out=None: einsum_impl('i,j->ij', input, vec2, out=out)),
    tensordot: Op(alias=lambda a, b, dims=2, out=None: einsum_impl(
        dim_tensordot(a.shape, b.shape, dims), a, b, out=out)),

    # not included in the primtorch oplist

    # TODO: slice may need distributed implementation
    torch.ops.aten.slice: Op(elwise=AllDBut('dim', default=0)),
}


def assert_call_eq(fn1: Callable, fn2: Callable, *args, **kw):
    assert_close(fn1(*args, **kw), fn2(*args, **kw))


assert_call_eq(
    ops[addbmm].alias, torch.addbmm,
    torch.randn(3,5), torch.randn(2,3,4), torch.randn(2,4,5), alpha=0.2, beta=0.8)
#), torch.addbmm(n, a, b, alpha=0.2, beta=0.8))


assert_call_eq(ops[tensordot].alias, torch.tensordot, torch.randn(2,3,4), torch.randn(3,4,5))