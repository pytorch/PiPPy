from argparse import ArgumentError
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from sympy import O

from spmd.tensor.placement_types import PlacementSpec
import itertools
import functools
import operator
from spmd.tensor.api import _Partial, Replicate, Shard

prod = lambda xs: functools.reduce(operator.mul, xs, 1)


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



def dim_repeat(shape, sizes):
    sizes = normalize_sizes(*sizes)
    assert len(sizes) >= len(shape), f'Number of dimensions of repeat dims {sizes} can not be smaller than number of dimensions of tensor {shape}.'
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
    to_size = infer_size(from_nelem, normalize_sizes(*to_size))

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
    if dim == -1:
        dim = len(dims)
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
    'Tensor.expand': Op(dim_map=lambda self, *sizes: expand(self.shape, normalize_sizes(*sizes)), shape_argnum=1),
    torch.flatten: Op(dim_map=lambda tensor: dim_flatten(tensor.shape)),
    torch.movedim: Op(dim_map=lambda input, source, destination: dim_movedim(len(input.shape), source, destination)),
    torch.permute: Op(dim_map=lambda input, dims: dims),
    torch.ravel: Op(dim_map=lambda tensor: dim_flatten(tensor.shape)),
    'Tensor.repeat': Op(dim_map=lambda self, *sizes: dim_repeat(self.shape, sizes)),
    torch.reshape: Op(dim_map=lambda input, shape: view_groups(input.shape, shape)[1], shape_argnum=1),
    torch.tile: Op(dim_map=lambda input, dims: dim_tile(input.shape, dims)),
    torch.transpose: Op(dim_map=lambda input, dim0, dim1: dim_transpose(input.shape, dim0, dim1)),
    torch.unsqueeze: Op(dim_map=lambda input, dim: dim_unsqueeze(input.shape, dim)),
    'Tensor.view': Op(dim_map=lambda input, *shape: view_groups(input.shape, shape)[1], shape_argnum=1),
}




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

import torch.distributed as dist
from spmd import DTensor, DeviceMesh, Shard, Replicate, distribute_tensor

import spmd
from spmd.tensor.dispatch import OpSchema

from spmd.tensor.ops.utils import register_prop_rule



def register_prop_rule_map(aten_op_name, local_op_name):

    @register_prop_rule(aten_op_name)
    def reshape_prop(op_schema: OpSchema):
        spec = ops[local_op_name]

        # note we are passing _global_ tensors
        rules = spec.dim_map(*op_schema.args, **op_schema.kwargs)

        # note we are passing _local_ tensor shapes
        local_out_shape, shard_out = propagate_shape_and_sharding(
            op_schema.args_spec[0].placements,
            op_schema.args_with_local_tensor[0].shape,
            rules,
            op_schema.args_spec[0].mesh.mesh.shape,
        )

        args  = op_schema.args_with_local_tensor
        if spec.shape_argnum is not None:
            op_schema.args_with_local_tensor = args[:spec.shape_argnum] + (tuple(local_out_shape), ) + args[spec.shape_argnum + 1:]

        return PlacementSpec(ndim=len(local_out_shape), mesh=op_schema.args_spec[0].mesh, placements=shard_out)


register_prop_rule_map('aten.view.default', 'Tensor.view')
register_prop_rule_map('aten.unsqueeze.default', torch.unsqueeze)
register_prop_rule_map('aten.expand.default', 'Tensor.expand')
register_prop_rule_map('aten.permute.default', torch.permute)
register_prop_rule_map('aten.repeat.default', 'Tensor.repeat')
register_prop_rule_map('aten.transpose.int', torch.transpose)


def test_call_dt(op_name, args, kwargs, should_throw: bool, device_mesh: DeviceMesh):
    spmd.tensor.dispatch._DEBUG_STRICT = True
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
            if dist.get_rank() == 0:
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

            sharding_choices = [Replicate()] + [
                    Shard(i) for i, s in enumerate(in_shape) if s > 1 and i not in no_shard_dims]

            all_sharding_choices = itertools.product(*(device_mesh.ndim * [sharding_choices]))

            # pick random shardings and broadcast them to all ranks
            # choice_idxs = torch.tensor([random.randrange(len(sharding_choices)) for _ in range(device_mesh.ndim)])
            # dist.broadcast(choice_idxs, src=0)

            # some random in_shard
            #in_shard = [sharding_choices[idx] for idx in choice_idxs]
            for in_shard in all_sharding_choices:
                in_dt = distribute_tensor(args[0], device_mesh, in_shard)
                out_dt = op(in_dt, *args[1:], **kwargs)

                full_out = out_dt.redistribute(device_mesh, device_mesh.ndim * [Replicate()]).to_local()

                if dist.get_rank() == 0:
                    assert outputs.shape == full_out.shape, f'Expected shape {outputs.shape}, got {full_out.shape}'
                    assert torch.allclose(outputs, full_out)

    except Exception as e:
        if not should_throw:
            print('------- expected call not to throw but got error -------')
            print(op)
            print(args)
            print(kwargs)
            raise e


def assert_throw(fn):
    try:
        fn()
    except:
        return
    assert False, 'didnt throw'


from torch import rand, randn


DEVICE_MESH = None

def test_dimmap(op, args, expected_rule_output):
    assert(ops[op].dim_map(*args) == expected_rule_output)
    # test_call(op, args, {}, False)
    test_call_dt(op, args, {}, False, DEVICE_MESH)
 

def dimmap_tests():
    test_dimmap(torch.atleast_1d, (randn(()), ), (SINGLETON, ))
    test_dimmap(torch.atleast_1d, (randn(24), ), (0, ))
    test_dimmap(torch.atleast_1d, (randn(24, 36), ), (0, 1))

    test_dimmap(torch.atleast_2d, (randn(()), ), (SINGLETON, SINGLETON))
    test_dimmap(torch.atleast_2d, (randn(24), ), (SINGLETON, 0))
    test_dimmap(torch.atleast_2d, (randn(24, 36), ), (0, 1))
    test_dimmap(torch.atleast_2d, (randn(24, 36, 48), ), (0, 1, 2))

    test_dimmap(torch.atleast_3d, (randn(()), ), (SINGLETON, SINGLETON, SINGLETON))
    test_dimmap(torch.atleast_3d, (randn(24), ), (SINGLETON, 0, SINGLETON))
    test_dimmap(torch.atleast_3d, (randn(24, 36), ), (0, 1, SINGLETON))
    test_dimmap(torch.atleast_3d, (randn(24, 36, 42), ), (0, 1, 2))
    test_dimmap(torch.atleast_3d, (randn(24, 36, 42, 24), ), (0, 1, 2, 3))

    assert_throw(lambda: ops[torch.broadcast_to].dim_map(randn(24, 36), (1,2,4)))

    test_dimmap(torch.broadcast_to, (rand(24, 36), (1, 24, 36)), (SINGLETON, 0, 1))
    test_dimmap(torch.broadcast_to, (rand(24, 36), (42, 24, 36)), (BROADCAST(SINGLETON, 42), 0, 1))
    test_dimmap(torch.broadcast_to, (rand(24, 1, 36), (12, 24, 24, 36)), (BROADCAST(SINGLETON, 12), 0, BROADCAST(1, 24), 2))
    test_dimmap(torch.broadcast_to, (rand(24, 36), (-1, 36)), (0, 1))
    test_dimmap(torch.broadcast_to, (rand(24, 1, 36), (-1, 1, 36)), (0, 1, 2))


    test_dimmap(
        torch.broadcast_to, (randn(36, 1, 24), (12, 36, 42, 24)),
        (BROADCAST(SINGLETON, 12), 0, BROADCAST(1, 42), 2)
    )

    test_dimmap(
        'Tensor.expand', (randn(24, 1, 36, 1), 36, 24, 42, -1, 24),
        (BROADCAST(SINGLETON, 36), 0, BROADCAST(1, 42), 2, BROADCAST(3, 24))
    )

    test_dimmap(
        'Tensor.expand', (randn(24, 1, 36, 1), (36, 24, 42, -1, 24)),
        (BROADCAST(SINGLETON, 36), 0, BROADCAST(1, 42), 2, BROADCAST(3, 24))
    )

    test_dimmap(torch.flatten, (randn(24, 36), ), (FLATTEN((0, 1)), ))
    test_dimmap(torch.flatten, (randn(42), ), (0, ))
    test_dimmap(torch.flatten, (randn(()), ), (SINGLETON, ))


    test_dimmap(torch.movedim, (randn(12, 24, 48, 96), 1, 2), (0, 2, 1, 3))
    test_dimmap(torch.movedim, (randn(6, 12, 24), 1, 0), (1, 0, 2))
    test_dimmap(torch.movedim, (randn(24, 12, 6), (1, 2), (0, 1)), (1, 2, 0))
    test_dimmap(torch.movedim, (randn(24, 6, 12), (0, 2, 1), (2, 1, 0)), (1, 2, 0))
    test_dimmap(torch.movedim, (randn(24, 12), (1, 0), (0, 1)), (1, 0))


    test_dimmap(torch.movedim, (randn(36, 24, 12), (1, 2), (0, 1)), (1, 2, 0))

    test_dimmap(torch.permute, (randn(24, 36, 42), (2, 0, 1)), (2, 0, 1))

    test_dimmap(torch.ravel, (randn(24, 36), ), (FLATTEN((0, 1)), ))
    test_dimmap(torch.ravel, (randn(42), ), (0, ))
    test_dimmap(torch.ravel, (randn(()), ), (SINGLETON, ))

    test_dimmap(
        'Tensor.repeat', (randn(24, 36), 1, 2, 1, 1, 2),
        (SINGLETON, BROADCAST(SINGLETON, 2), SINGLETON, 0, REPEAT(1, 2))
    )

    test_dimmap(
        torch.reshape, (randn(6, 12, 24), (72, 24)),
        (FLATTEN((0, 1)), 2),
    )

    test_dimmap(
        torch.tile, (randn(24, 36), (1, 2, 1, 1, 2)),
        (SINGLETON, BROADCAST(SINGLETON, 2), SINGLETON, 0, REPEAT(1, 2))
    )
    test_dimmap(
        torch.tile, (randn(42, 24, 36), (1, 3, )),
        (0, 1, REPEAT(2, 3)),
    )

    test_dimmap(torch.transpose, (randn(24, 60, 42, 60), 2, 0), (2, 1, 0, 3))

    test_dimmap(torch.unsqueeze, (randn(42, 24, 36), 1), (0, SINGLETON, 1, 2))

    test_dimmap(
        'Tensor.view', (randn(6, 12, 24), 72, 24),
        (FLATTEN((0, 1)), 2),
    )

    test_dimmap(
        'Tensor.view', (randn(1, 1, 12), -1),
        (2, ),
    )

    test_dimmap(
        'Tensor.view', (randn(1, 1, 42, 24), -1),
        (FLATTEN((2, 3)), ),
    )

    test_dimmap(
        'Tensor.view', (randn(1, 1, 42, 1, 24, 1), -1),
        (FLATTEN((2, 4)), ),
    )


if __name__ == '__main__':
    dist.init_process_group('gloo')
    DEVICE_MESH = DeviceMesh('cpu', torch.arange(dist.get_world_size()).view(-1, 2))
    dimmap_tests()