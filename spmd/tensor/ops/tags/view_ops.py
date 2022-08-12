from argparse import ArgumentError
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


# output is new singleton dimension
SINGLETON = ('singleton', )


def BROADCAST(input_dim, dim_size):
    """
    output is the broadcast of a singleton input dimension
    """
    return ('broadcast', input_dim, dim_size)


def NEWDIM(val):
    """
    create a new 'generic' dimension with a given size 'val'
    """
    if val == 1:
        # a NEWDIM of size 1 is a singleton!
        return SINGLETON
    else:
        return ('newdim', val)


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
        return ('comp', input_dim, new_group_shape, new_idx)


def dim_pad_left(input_shape, min_dims):
    return (SINGLETON, ) * max(0, min_dims - len(input_shape)) + tuple(range(len(input_shape)))


def dim_atleast_3d(num_dims):
    if num_dims == 0:
        return (SINGLETON, SINGLETON, SINGLETON)
    elif num_dims == 1:
        return (SINGLETON, 0, SINGLETON)
    elif num_dims == 2:
        return (0, 1, SINGLETON)
    else:
        return tuple(range(num_dims))


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


def normalize_sizes(*sizes):
    if isinstance(sizes[0], int):
        return sizes
    elif len(sizes) == 1:
        return sizes[0]
    else:
        raise ArgumentError('Size must be int... or tuple')


def dim_flatten(shape):
    if len(shape) == 0:
        return (SINGLETON, )
    elif len(shape) == 1:
        return (0, )
    else:
        return (FLATTEN(tuple(range(len(shape)))), )


def dim_movedim(shape_len, input, destination):
    if not isinstance(input, tuple):
        input = (input, )
    if not isinstance(destination, tuple):
        destination = (destination, )
    
    assert len(input) == len(destination)
    input_set = set(input)
    assert len(input_set) == len(input), 'Found repeated input dims'
    assert len(set(destination)) == len(destination), 'Found repeated output dims'
    assert max(input) < shape_len
    assert max(destination) < shape_len

    dest = [-1, ] * shape_len
    for i, d in zip(input, destination):
        dest[d] = i

    unused_inputs_iter = iter(i for i in range(shape_len) if i not in input_set)
    for i in range(shape_len):
        if dest[i] == -1:
            dest[i] = next(unused_inputs_iter)

    return tuple(dest)


import functools
import operator


prod = lambda xs: functools.reduce(operator.mul, xs, 1)


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

    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)

    result = []
    result_dim = []

    while from_idx < from_len or to_idx < to_len:
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
                    nf = from_size[from_idx]

                    from_group.append(nf)
                    from_group_dim.append(from_idx)

                    from_idx += 1
                    f *= nf
                else:
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
            result_pp += [COMP(ff, tr, i) for i in range(len(r))]
        result_pp = tuple(result_pp)

    return result, result_pp


def dim_tile(shape, dims):
    if len(dims) < len(shape):
        dims = (1, ) * (len(shape) - len(dims)) + dims
    return dim_repeat(shape, dims)


def dim_transpose(shape, dim1, dim2):
    assert dim1 < len(shape)
    assert dim2 < len(shape)
    shape = list(range(len(shape)))
    shape[dim1] = dim2
    shape[dim2] = dim1
    return tuple(shape)


def dim_unsqueeze(shape, dim):
    dims = tuple(range(len(shape)))
    return dims[:dim] + (SINGLETON, ) + dims[dim:]


@dataclass
class Op:
    dim_map: Optional[List[int]] = None
    shape_argnum: Optional[Union[int, str]] = None
    inputwise: bool = False   # whether the op operates on each of its input independently, in which case
                              # lambdas will be called per-input
 


import torch
ops = {
    torch.atleast_1d: Op(inputwise=True, dim_map=lambda x: dim_pad_left(x.shape, 1)),
    torch.atleast_2d: Op(inputwise=True, dim_map=lambda x: dim_pad_left(x.shape, 2)),
    torch.atleast_3d: Op(inputwise=True, dim_map=lambda x: dim_atleast_3d(len(x.shape))),
    torch.broadcast_to: Op(dim_map=lambda input, shape: expand(input.shape, shape), shape_argnum=1),
    'Tensor.expand': Op(dim_map=lambda self, *sizes: expand(self.shape, normalize_sizes(*sizes)), shape_argnum='*'),
    torch.flatten: Op(dim_map=lambda tensor: dim_flatten(tensor.shape)),
    torch.movedim: Op(dim_map=lambda input, source, destination: dim_movedim(len(input.shape), source, destination)),
    torch.permute: Op(dim_map=lambda input, dims: dims),
    torch.ravel: Op(dim_map=lambda tensor: dim_flatten(tensor.shape)),
    'Tensor.repeat': Op(dim_map=lambda self, *sizes: dim_repeat(self.shape, sizes)),
    torch.reshape: Op(dim_map=lambda input, shape: view_groups(input.shape, shape)[1], shape_argnum=1),
    torch.tile: Op(dim_map=lambda input, dims: dim_tile(input.shape, dims)),
    torch.transpose: Op(dim_map=lambda input, dim0, dim1: dim_transpose(input.shape, dim0, dim1)),
    torch.unsqueeze: Op(dim_map=lambda input, dim: dim_unsqueeze(input.shape, dim)),
    'Tensor.view': Op(dim_map=lambda input, *shape: view_groups(input.shape, shape)[1], shape_argnum='*'),
}



print('------------------ Sharding rules -------------------')


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


print('------------------ Sharding and shape propagation -------------')


def propagate_shape_and_sharding(in_shard, local_in_shape, rule, mesh_sizes):
    """
    Takes as input the shape of the _local_ tensor, and the input sharding,
    and produce corresponding output sharding and shape of the _local_ output tensor.
    """
    assert len(in_shard) == len(mesh_sizes)
    # print('local_output_shape:', in_shard, local_in_shape, rule, mesh_sizes)

    sharded_in_dims = set(s.dim for s in in_shard if isinstance(s, Shard))

    def get_dim_size(cmd):
        if isinstance(cmd, int):
            return (
                local_in_shape[cmd],
                cmd if cmd in sharded_in_dims else None
            )

        elif cmd[0] == 'flatten':
            for in_dim in cmd[1][1:]:
                assert not in_dim in sharded_in_dims, 'Only the first member of a FLATTEN group can be sharded'
            return (
                prod(get_dim_size(a)[0] for a in cmd[1]),
                cmd[1][0] if cmd[1][0] in sharded_in_dims else None
            )
        elif cmd[0] == 'comp':
            if cmd[2] > 0:
                # we will shard only on the first dimension of the group
                # so the shape should be the nominal shape of the component
                return cmd[1][cmd[2]], None
            else:
                dim_size, in_dim = get_dim_size(cmd[1])
                if in_dim is None:
                    # in case input dim is not sharded; our size will be
                    # the size of the corresponding input
                    return dim_size, None
                # we need to check that the input dimension is divisble
                # by the size of the submesh we're sharding it on
                submesh_size = 1
                for size, shard in zip(mesh_sizes, in_shard):
                    if isinstance(shard, Shard) and shard.dim == in_dim:
                        submesh_size *= size
                out_size = cmd[1][0]
                assert out_size % submesh_size, 'Resulting dimension size is not divisible by its mesh dimension.'
                return out_size // submesh_size, in_dim
        elif cmd[0] == 'singleton':
            return 1, None
        elif cmd[0] == 'broadcast':
            return cmd[2], None
        elif cmd[0] == 'newdim':
            return cmd[1], None
        elif cmd[0] == 'repeat':
            assert not cmd[1] in sharded_in_dims, 'Cannot tile sharded dimension.'
            return local_in_shape[cmd[1]], None
        else:
            raise RuntimeError(f'cmd not found: {cmd}, in rule: {rule}')
 
    dim_map = {}
    out_shape = []
    for dim, cmd in enumerate(rule):
        out_size, in_dim = get_dim_size(cmd)
        out_shape.append(out_size)
        if in_dim is not None:
            dim_map[in_dim] = dim

    return (
        out_shape,
        [Shard(dim_map[s.dim]) if isinstance(s, Shard) else s for s in in_shard]
    )


print('-----------------  TiledTensor ----------------------')


import numpy as np


class TiledTensor:
    def __init__(self, mesh, spec, tiles):
        self.mesh = mesh
        self.spec = spec
        self.tiles = tiles

    def remove_last(self):
        """ return TiledTensor with one less mesh dimension """
        new_mesh = self.mesh[:-1]
        new_spec = self.spec[:-1]
        new_tiles = np.empty(new_mesh, dtype=object)
        if isinstance(self.spec[-1], Shard):

            def scat(a, dim):
                x = np.empty((), dtype=object)
                x.itemset(torch.cat(tuple(a), dim=dim))
                return x

            new_tiles = np.apply_along_axis(
                lambda x: scat(x, dim=self.spec[-1].dim),
                -1, self.tiles)
        elif isinstance(self.spec[-1], _Partial):
            from functools import reduce
            new_tiles = np.apply_along_axis(lambda x: reduce(torch.add, x), -1, self.tiles)
        elif isinstance(self.spec[-1], Replicate):
            new_tiles = self.tiles[..., 0]
            for i in range(self.tiles.shape[-1]):
                for a, b in zip(self.tiles[..., i].flatten(), new_tiles.flatten()):
                    assert(torch.allclose(a,b))
        else:
            raise RuntimeError('Unknown shard type.')
        t = TiledTensor(new_mesh, new_spec, new_tiles)
        return t

    def add_dim(self, spec, size):
        new_mesh = self.mesh + [size]
        new_spec = self.spec + [spec]
        if isinstance(spec, Shard):
            flat_tiles = self.tiles.reshape(-1)
            new_tiles = np.empty(new_mesh, dtype=object)
            new_flat_tiles = new_tiles.reshape(-1, new_tiles.shape[-1])
            for i in range(flat_tiles.shape[0]):
                chunks = torch.chunk(flat_tiles[i], size, dim=spec.dim)
                if len(chunks) < size:
                    # special case when chunk returns less than the requested size.
                    # e.g. if we are sharding a tensor of shape [1] on 3 shards, we'll end up with 2 empty shards
                    # the code below guarantees that we actually get that.
                    t = chunks[0]
                    empty_shape = t.shape[:spec.dim] + (0, ) + t.shape[spec.dim + 1:]
                    chunks += (size - len(chunks)) * (torch.empty(empty_shape, dtype=t.dtype), )
                new_flat_tiles[i, :] = chunks
        elif isinstance(spec, Replicate):
            new_tiles = np.expand_dims(self.tiles, -1).repeat(size, -1)
        else:
            raise RuntimeError('Unknown shard type.')
        return TiledTensor(new_mesh, new_spec, new_tiles)

    def __repr__(self):
        return f'mesh={repr(self.mesh)}, spec={repr(self.spec)}, tiles={repr(self.tiles)}'

    def to_full(self):
        if len(self.mesh) == 0:
            return self.tiles.item()
        else:
            return self.remove_last().to_full()


def to_tiled(mesh_sizes, spec, full):
    tile = np.empty((), dtype=object)
    tile.itemset(full)
    tiled = TiledTensor([], [], tile)
    for dim_size, dim_spec in zip(mesh_sizes, spec):
        tiled = tiled.add_dim(dim_spec, dim_size)
    return tiled


import itertools


def test_tiling(full, mesh_dims=None):
    if mesh_dims == None:
        mesh_dims = [2, 2]

    shard_options = [Replicate()] + [Shard(i) for i in range(len(full.shape))]
    possibilities = list(itertools.product(*(len(mesh_dims) * [shard_options])))

    for spec in possibilities:
        tiled = to_tiled(mesh_dims, spec, full)
        rfull = tiled.to_full()
        assert(full.shape == rfull.shape)
        assert(torch.allclose(full, rfull))


test_tiling(torch.rand(1), [3])
test_tiling(torch.rand(3), [2])
test_tiling(torch.rand((4, 3)))
test_tiling(torch.rand((8,8)))
test_tiling(torch.rand((3,1)))
test_tiling(torch.rand(1,1))


print('------------- Run tiled ops for test purposes -------------')


def run_tiled(in_tiled: TiledTensor, full_shape, op, spec, args, kwargs, rules):
    in_shard = in_tiled.spec
    mesh_sizes = in_tiled.mesh
    _, out_shard = propagate_shape_and_sharding(in_shard, full_shape, rules, mesh_sizes)

    out_tiles_flat = np.empty(in_tiled.tiles.size, dtype=object)
    for i, in_tile in enumerate(in_tiled.tiles.flatten()):
        if spec.shape_argnum is not None:
            local_out_shape, _ = propagate_shape_and_sharding(in_shard, in_tile.shape, rules, mesh_sizes)
            if spec.shape_argnum == '*':
                # HACK ,assume first input is tensor, others shape (view)
                args = [args[0]] + list(local_out_shape)
            elif isinstance(spec.shape_argnum, int):
                args = list(args)
                args[spec.shape_argnum] = tuple(local_out_shape)
            else:
                raise RuntimeError('Invalid shape_argnum.')

        # HACK, assume first inpt is tensor, others not.
        out_tile = op(in_tile, *args[1:], **kwargs)
        out_tiles_flat.itemset(i, out_tile)

    out_tiles = out_tiles_flat.reshape(in_tiled.tiles.shape)

    return TiledTensor(mesh_sizes, out_shard, out_tiles)


from torch.utils._pytree import tree_flatten
import random


def get_op_func(op_name):
    if isinstance(op_name, str):
        splat = op_name.split('.')
        if len(splat) == 2:
            cls, meth = splat
            assert cls == 'Tensor'
            return getattr(torch.Tensor, meth)
        else:
            assert len(splat) == 1
            return getattr(torch.Tensor, splat[0])
    else:
        return op_name



def test_call(op_name, args, kwargs, should_throw):
    spec = ops[op_name]
    op = get_op_func(op_name)

    try:
        rules = spec.dim_map(*args, **kwargs)
        outputs = op(*args, **kwargs)
        assert not should_throw

        flat_args, _ = tree_flatten(args)
        output_shapes = [a.shape for a in outputs] if isinstance(outputs, tuple) else outputs.shape

        if isinstance(rules, list):
            print('not testing this one: ', op)
            assert len(flat_args) == len(rules)
            # expected = [expected_shape(arg.shape, rule) for arg, rule in zip(flat_args, rules)]
        else:
            print('-------- ', op)
            # print(rules)
            in_shape = flat_args[0].shape

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
            
            if op == torch.unbind:
                no_shard_dims.add(kwargs.get('dim', 0))

            # pick some random mesh size
            # total size of mesh shouldn't be larger than size of the input tensor
            input = args[0]
            if input.numel() >= 8:
                mesh_sizes = [2, 4]
            elif input.numel() >= 4:
                mesh_sizes = [2, 2]
            elif input.numel() >= 2:
                mesh_sizes = [2]
            else:
                mesh_sizes = [1]

            # print('mesh sizes based on input shape: ', mesh_sizes)

            def pick_sharding(in_shape, mesh_dim_size):
                # - do not shard on singleton dimensions (this wouldn't happen assuming DT is well constructed)
                # - only shard on first dim of flattened groups
                # leaving partial out for now.
                return random.choice([Replicate()] + [
                    Shard(i) for i, s in enumerate(in_shape) if s > 1 and i not in no_shard_dims]
                )

            # some random in_shard
            in_shard = [pick_sharding(in_shape, d) for d in mesh_sizes]
            in_tiled = to_tiled(mesh_sizes, in_shard, args[0])
            out_tiled = run_tiled(in_tiled, in_shape, op, spec, args, kwargs, rules) 
            full_out = out_tiled.to_full()

            assert(outputs.shape == full_out.shape)
            assert(torch.allclose(outputs, full_out))

    except Exception as e:
        if not should_throw:
            print('------- expected call not to throw but got error -------')
            print(op)
            print(args)
            print(kwargs)
            raise e


print('------------ unit testing ------------')


def assert_throw(fn):
    try:
        fn()
    except:
        return
    assert False, 'didnt throw'



from torch import rand, randn


def test_dimmap(op, args, expected_rule_output):
    assert(ops[op].dim_map(*args) == expected_rule_output)
    test_call(op, args, {}, False)
 


test_dimmap(torch.atleast_1d, (randn(()), ), (SINGLETON, ))
test_dimmap(torch.atleast_1d, (randn(2), ), (0, ))
test_dimmap(torch.atleast_1d, (randn(2, 3), ), (0, 1))

test_dimmap(torch.atleast_2d, (randn(()), ), (SINGLETON, SINGLETON))
test_dimmap(torch.atleast_2d, (randn(2), ), (SINGLETON, 0))
test_dimmap(torch.atleast_2d, (randn(2, 3), ), (0, 1))
test_dimmap(torch.atleast_2d, (randn(2, 3, 4), ), (0, 1, 2))

test_dimmap(torch.atleast_3d, (randn(()), ), (SINGLETON, SINGLETON, SINGLETON))
test_dimmap(torch.atleast_3d, (randn(2), ), (SINGLETON, 0, SINGLETON))
test_dimmap(torch.atleast_3d, (randn(2, 3), ), (0, 1, SINGLETON))
test_dimmap(torch.atleast_3d, (randn(2, 3, 4), ), (0, 1, 2))
test_dimmap(torch.atleast_3d, (randn(2, 3, 4, 2), ), (0, 1, 2, 3))

assert_throw(lambda: ops[torch.broadcast_to].dim_map(randn(2,3), (1,2,4)))

test_dimmap(torch.broadcast_to, (rand(2,3), (1,2,3)), (SINGLETON, 0, 1))
test_dimmap(torch.broadcast_to, (rand(2,3), (4,2,3)), (BROADCAST(SINGLETON, 4), 0, 1))
test_dimmap(torch.broadcast_to, (rand(2,1,3), (2,2,2,3)), (BROADCAST(SINGLETON, 2), 0, BROADCAST(1, 2), 2))
test_dimmap(torch.broadcast_to, (rand(2,3), (-1,3)), (0, 1))
test_dimmap(torch.broadcast_to, (rand(2,1,3), (-1,1,3)), (0, 1, 2))


test_dimmap(
    torch.broadcast_to, (randn(3,1,2), (2,3,4,2)),
    (BROADCAST(SINGLETON, 2), 0, BROADCAST(1, 4), 2)
)

test_dimmap(
    'Tensor.expand', (randn(2,1,3,1), 3,2,4,-1,2),
    (BROADCAST(SINGLETON, 3), 0, BROADCAST(1, 4), 2, BROADCAST(3, 2))
)

test_dimmap(
    'Tensor.expand', (randn(2,1,3,1), (3,2,4,-1,2)),
    (BROADCAST(SINGLETON, 3), 0, BROADCAST(1, 4), 2, BROADCAST(3, 2))
)

test_dimmap(torch.flatten, (randn(2,3), ), (FLATTEN((0, 1)), ))
test_dimmap(torch.flatten, (randn(4), ), (0, ))
test_dimmap(torch.flatten, (randn(()), ), (SINGLETON, ))


test_dimmap(torch.movedim, (randn(4, 8, 16, 32), 1, 2), (0, 2, 1, 3))
test_dimmap(torch.movedim, (randn(2, 4, 8), 1, 0), (1, 0, 2))
test_dimmap(torch.movedim, (randn(8, 4, 2), (1, 2), (0, 1)), (1, 2, 0))
test_dimmap(torch.movedim, (randn(8, 2, 4), (0, 2, 1), (2, 1, 0)), (1, 2, 0))
test_dimmap(torch.movedim, (randn(8, 4), (1, 0), (0, 1)), (1, 0))


test_dimmap(torch.movedim, (randn(3,2,1), (1, 2), (0, 1)), (1, 2, 0))

test_dimmap(torch.permute, (randn(2,3,4), (2, 0, 1)), (2, 0, 1))

test_dimmap(torch.ravel, (randn(2,3), ), (FLATTEN((0, 1)), ))
test_dimmap(torch.ravel, (randn(4), ), (0, ))
test_dimmap(torch.ravel, (randn(()), ), (SINGLETON, ))

test_dimmap(
    'Tensor.repeat', (randn(2,3), 1, 2, 1, 1, 2),
    (SINGLETON, BROADCAST(SINGLETON, 2), SINGLETON, 0, REPEAT(1, 2))
)

test_dimmap(
    torch.reshape, (randn(2,4,8), (8, 8)),
    (FLATTEN((0, 1)), 2),
)

test_dimmap(
    torch.tile, (randn(2,3), (1, 2, 1, 1, 2)),
    (SINGLETON, BROADCAST(SINGLETON, 2), SINGLETON, 0, REPEAT(1, 2))
)
test_dimmap(
    torch.tile, (randn(4, 2,3), (1, 3, )),
    (0, 1, REPEAT(2, 3)),
)

test_dimmap(torch.transpose, (randn(2,5,4,5), 2, 0), (2,1,0,3))

test_dimmap(torch.unsqueeze, (randn(4,2,3), 1), (0, SINGLETON, 1, 2))

test_dimmap(
    'Tensor.view', (randn(2,4,8), 8, 8),
    (FLATTEN((0, 1)), 2),
)

test_dimmap(
    'Tensor.view', (randn(1, 1, 4), -1),
    (2, ),
)

test_dimmap(
    'Tensor.view', (randn(1, 1, 4, 2), -1),
    (FLATTEN((2, 3)), ),
)

test_dimmap(
    'Tensor.view', (randn(1, 1, 4, 1, 2, 1), -1),
    (FLATTEN((2, 4)), ),
)