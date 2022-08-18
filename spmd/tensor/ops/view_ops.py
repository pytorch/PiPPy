from argparse import ArgumentError
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Set,
    Union,
    Sequence,
    cast,
)

from spmd.tensor.placement_types import DTensorSpec, Placement
import functools
import operator
from spmd.tensor.api import Shard
from spmd.tensor.dispatch import OpSchema, OutputSharding
from spmd.tensor.ops.utils import register_prop_rule


def _prod(xs: Iterable[int]) -> int:
    return functools.reduce(operator.mul, xs, 1)


Shape = Tuple[int, ...]
Singleton = Tuple[str]  # This is always ('singleton', )
InputDim = int  # The output dimension corresponds to the i-th input dimension
NewDim = Tuple[str, int]  # ('newdim', new_dim_size)

# NOTE: Could not figure out how to make pyre work with forward references
#       beyond self-reference, so I am duplicating the definition below
DimSpec = Union[
    InputDim,
    Singleton,
    Tuple[str, "DimSpec", int],  # Broadcast
    NewDim,
    Tuple[str, "DimSpec", int],  # Repeat
    Tuple[str, Tuple["DimSpec", ...]],  # Flatten
    Tuple[str, "DimSpec", Shape, int],  # Comp
]
Broadcast = Tuple[str, DimSpec, int]  #  ('broadcast', input_dim_spec, size)
Repeat = Tuple[str, DimSpec, int]  # ('repeat', input_dim_spec, times)
Flatten = Tuple[str, Tuple[DimSpec, ...]]  # ('flatten', (input_dim_spec, ...))
Comp = Tuple[
    str, DimSpec, Shape, int
]  # ('comp', dim_spec, group_shape, component_id_in_group)

# Rules that map each dimension of the output to dimensions of the input tensor
DimMap = Tuple[DimSpec, ...]


Dim = InputDim

SINGLETON: Singleton = ("singleton",)


def BROADCAST(input_dim: DimSpec, dim_size: int) -> Broadcast:
    """
    output is the broadcast of a singleton input dimension
    """
    return ("broadcast", input_dim, dim_size)


def NEWDIM(val: int) -> Union[NewDim, Singleton]:
    """
    create a new 'generic' dimension with a given size 'val'
    """
    if val == 1:
        # a NEWDIM of size 1 is a singleton!
        return SINGLETON
    else:
        return ("newdim", val)


def REPEAT(input_dim: DimSpec, times: int) -> DimSpec:
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
        return ("repeat", input_dim, times)


def FLATTEN(input_dims: Tuple[DimSpec, ...]) -> DimSpec:
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
        return ("flatten", input_dims)


def COMP(input_dim: DimSpec, group_shape: Tuple[int, ...], idx: int) -> DimSpec:
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
        group_mapping = list(
            enumerate((s, i) for i, s in enumerate(group_shape) if s != 1)
        )
        new_group_shape = tuple(m[1][0] for m in group_mapping)
        new_idx = next(filter(lambda x: x[1][1] == idx, group_mapping))[0]
        return ("comp", input_dim, new_group_shape, new_idx)


def dim_pad_left(ndim: int, min_dims: int) -> DimMap:
    return (SINGLETON,) * max(0, min_dims - ndim) + tuple(range(ndim))


def dim_atleast_3d(ndim: int) -> DimMap:
    if ndim == 0:
        return (SINGLETON, SINGLETON, SINGLETON)
    elif ndim == 1:
        return (SINGLETON, 0, SINGLETON)
    elif ndim == 2:
        return (0, 1, SINGLETON)
    else:
        return tuple(range(ndim))


def expand(input_shape: Shape, shape: Shape) -> DimMap:
    """Implements broadcast on multiple dimensions"""
    assert len(shape) >= len(input_shape)

    # 1. create padded input dimensions
    padded_input = dim_pad_left(len(input_shape), len(shape))
    # 2. check that input shapes are compatible
    mapping = []
    for p, desired_s in zip(padded_input, shape):
        if p == SINGLETON:
            actual_s = 1
            assert desired_s >= 0
        else:
            actual_s = input_shape[p]
            assert actual_s == 1 or desired_s == -1 or desired_s == actual_s
        mapping.append(
            p
            if desired_s in (1, -1) or desired_s == actual_s
            else BROADCAST(p, desired_s)
        )
    return tuple(mapping)


def normalize_sizes(sizes: Union[Shape, Tuple[Shape]]) -> Shape:
    if isinstance(sizes[0], int):
        return cast(Shape, sizes)
    elif len(sizes) == 1:
        return cast(Shape, sizes[0])
    else:
        raise RuntimeError("Size must be int... or tuple")


def dim_flatten(ndim: int) -> DimMap:
    if ndim == 0:
        return (SINGLETON,)
    elif ndim == 1:
        return (0,)
    else:
        return (FLATTEN(tuple(range(ndim))),)


def normalize_dims(dims: Union[int, Tuple[int, ...]], ndim: int) -> Tuple[int]:
    if isinstance(dims, int):
        dims = (dims,)
    return tuple(normalize_dim(dim, ndim) for dim in dims)


def normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def dim_movedim(
    ndim: int,
    input: Union[int, Tuple[int, ...]],
    destination: Union[int, Tuple[int, ...]],
) -> DimMap:
    input = normalize_dims(input, ndim)
    destination = normalize_dims(destination, ndim)

    assert len(input) == len(destination)
    input_set = set(input)
    assert len(input_set) == len(input), "Found repeated input dims"
    assert len(set(destination)) == len(
        destination
    ), "Found repeated output dims"
    assert max(input) < ndim
    assert max(destination) < ndim

    dest = [
        -1,
    ] * ndim
    for i, d in zip(input, destination):
        dest[d] = i

    unused_inputs_iter = iter(i for i in range(ndim) if i not in input_set)
    for i in range(ndim):
        if dest[i] == -1:
            dest[i] = next(unused_inputs_iter)

    return tuple(dest)


def dim_repeat(ndim: int, sizes: Shape) -> DimMap:
    sizes = normalize_sizes(sizes)
    assert (
        len(sizes) >= ndim
    ), f"Number of dimensions of repeat dims {sizes} can not be smaller than number of dimensions of tensor {ndim}."
    pad = len(sizes) - ndim
    return tuple(REPEAT(SINGLETON, s) for s in sizes[:pad]) + tuple(
        REPEAT(i, s) for i, s in enumerate(sizes[pad:])
    )


def infer_size(total_size: int, sizes: Shape) -> Shape:
    """
    One dimension input to view may be "-1".
    Infer the size of this dimension given the total_size.
    """
    infers = [i for i, s in enumerate(sizes) if s == -1]
    size = _prod(sizes)
    assert len(infers) <= 1, "can only infer one size"
    if infers:
        size = -size
        missing_size = total_size // size
        assert (
            total_size % size == 0
        ), f"size inferred for -1 is not integral {sizes} should have {total_size} elements."
        return tuple(s if s != -1 else missing_size for s in sizes)
    assert size == total_size, f"sizes do not match {total_size} vs {size}"
    return sizes


def view_groups(
    from_size: Shape, to_size: Shape
) -> Tuple[List[Tuple[List[int], List[int]]], DimMap]:
    """
    Split up the total view into smaller groups of dimensions whose size will match:
    view_groups([3, 4, 5], [12, 5]) -> [([3, 4], [12]), ([5], [5])]
    """
    from_nelem = _prod(from_size)
    to_size = infer_size(from_nelem, normalize_sizes(to_size))

    assert from_nelem == _prod(to_size), "Total view shape does not add up"

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


def dim_tile(ndim: int, dims: Tuple[int, ...]) -> DimMap:
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    assert dim1 < ndim
    assert dim2 < ndim
    dimmap = list(range(ndim))
    dimmap[dim1] = dim2
    dimmap[dim2] = dim1
    return tuple(dimmap)


def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    dims = tuple(range(ndim))
    if dim < 0:
        dim += ndim + 1
    return dims[:dim] + (SINGLETON,) + dims[dim:]


@dataclass
class Op:
    dim_map: Callable[..., DimMap]
    shape_argnum: Optional[int] = None


import torch
from torch import Tensor

ops: Dict[Callable[..., torch.Tensor], Op] = {
    torch.atleast_1d: Op(dim_map=lambda x: dim_pad_left(x.ndim, 1)),
    torch.atleast_2d: Op(dim_map=lambda x: dim_pad_left(x.ndim, 2)),
    torch.atleast_3d: Op(dim_map=lambda x: dim_atleast_3d(x.ndim)),
    torch.broadcast_to: Op(
        dim_map=lambda input, shape: expand(input.shape, shape), shape_argnum=1
    ),
    Tensor.expand: Op(
        dim_map=lambda self, *sizes: expand(self.shape, normalize_sizes(sizes)),
        shape_argnum=1,
    ),
    torch.flatten: Op(dim_map=lambda tensor: dim_flatten(tensor.ndim)),
    torch.movedim: Op(
        dim_map=lambda input, source, destination: dim_movedim(
            input.ndim, source, destination
        )
    ),
    torch.permute: Op(
        dim_map=lambda input, dims: normalize_dims(dims, input.ndim)
    ),
    torch.ravel: Op(dim_map=lambda tensor: dim_flatten(tensor.ndim)),
    Tensor.repeat: Op(
        dim_map=lambda self, *sizes: dim_repeat(self.ndim, sizes)
    ),
    torch.reshape: Op(
        dim_map=lambda input, shape: view_groups(input.shape, shape)[1],
        shape_argnum=1,
    ),
    torch.tile: Op(dim_map=lambda input, dims: dim_tile(input.ndim, dims)),
    torch.transpose: Op(
        dim_map=lambda input, dim0, dim1: dim_transpose(input.ndim, dim0, dim1)
    ),
    torch.unsqueeze: Op(
        dim_map=lambda input, dim: dim_unsqueeze(input.ndim, dim)
    ),
    Tensor.view: Op(
        dim_map=lambda input, *shape: view_groups(input.shape, shape)[1],
        shape_argnum=1,
    ),
}


def propagate_shape_and_sharding(
    in_shard: Sequence[Placement],
    local_in_shape: Shape,
    rule: DimMap,
    mesh_sizes: Shape,
) -> Tuple[Shape, Sequence[Placement]]:
    """
    Takes as input the shape of the _local_ tensor, and the input sharding,
    and produce corresponding output sharding and shape of the _local_ output tensor.
    """
    assert len(in_shard) == len(mesh_sizes)
    # print('local_output_shape:', in_shard, local_in_shape, rule, mesh_sizes)

    sharded_in_dims: Set[int] = set(
        s.dim for s in in_shard if isinstance(s, Shard)
    )

    def get_dim_size(cmd: DimSpec) -> Tuple[int, Optional[InputDim]]:
        if isinstance(cmd, int):
            return (
                local_in_shape[cmd],
                cmd if cmd in sharded_in_dims else None,
            )

        elif cmd[0] == "flatten":
            cmd = cast(Flatten, cmd)
            for in_dim in cmd[1][1:]:
                assert (
                    not in_dim in sharded_in_dims
                ), "Only the first member of a FLATTEN group can be sharded"
            return (
                _prod(get_dim_size(a)[0] for a in cmd[1]),
                cast(InputDim, cmd[1][0])
                if cmd[1][0] in sharded_in_dims
                else None,
            )
        elif cmd[0] == "comp":
            cmd = cast(Comp, cmd)
            if cmd[3] > 0:
                # we will shard only on the first dimension of the group
                # so the shape should be the nominal shape of the component
                return cmd[2][cmd[3]], None
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
                out_size = cmd[2][0]
                assert (
                    out_size % submesh_size == 0
                ), f"Resulting dimension size {out_size} is not divisible by its mesh dimension {submesh_size}."
                return out_size // submesh_size, in_dim
        elif cmd[0] == "singleton":
            return 1, None
        elif cmd[0] == "broadcast":
            return cast(Broadcast, cmd)[2], None
        elif cmd[0] == "newdim":
            return cast(NewDim, cmd)[1], None
        elif cmd[0] == "repeat":
            cmd = cast(Repeat, cmd)
            size, in_dim = get_dim_size(cmd[1])
            assert (
                in_dim not in sharded_in_dims
            ), "Cannot tile sharded dimension."
            return size * cmd[2], None
        else:
            raise RuntimeError(f"cmd not found: {cmd}, in rule: {rule}")

    dim_map = {}
    out_shape = []
    for dim, cmd in enumerate(rule):
        out_size, in_dim = get_dim_size(cmd)
        out_shape.append(out_size)
        if in_dim is not None:
            dim_map[in_dim] = dim

    return (
        tuple(out_shape),
        [
            Shard(dim_map[s.dim]) if isinstance(s, Shard) else s
            for s in in_shard
        ],
    )


def local_shape(spec: DTensorSpec, rank: int) -> Tuple[int, ...]:
    """
    Given a DTensorSpec and a global rank, compute the shape of a local
    shard of the given DTensor.
    """
    assert spec.shape is not None, "DTensorSpec does not contain global shape."
    local_shape = list(spec.shape)  # start with global shape
    for idx, placement in enumerate(spec.placements):
        if isinstance(placement, Shard):
            assert (
                local_shape[placement.dim] % spec.mesh.size(idx) == 0
            ), "Only even sharding supported for now."
            local_shape[placement.dim] //= spec.mesh.size(idx)
    return tuple(local_shape)


def register_prop_rule_map(
    aten_op_name: str, local_op_name: Callable[..., torch.Tensor]
) -> None:
    @register_prop_rule(aten_op_name)
    def reshape_prop(op_schema: OpSchema) -> OutputSharding:
        spec = ops[local_op_name]

        # note we are passing _global_ tensors
        rules = spec.dim_map(*op_schema.args_schema, **op_schema.kwargs_schema)

        if torch.distributed.get_rank() == 0:
            print("----", aten_op_name)
            print(rules)
            print(op_schema.args_schema)

        # note we are passing _local_ tensor shapes
        input_dtensor_spec = op_schema.args_schema[0]
        assert isinstance(
            input_dtensor_spec, DTensorSpec
        ), "Expected first input to be a DTensorSpec"
        local_out_shape, shard_out = propagate_shape_and_sharding(
            input_dtensor_spec.placements,
            local_shape(input_dtensor_spec, torch.distributed.get_rank()),
            rules,
            tuple(input_dtensor_spec.mesh.mesh.shape),
        )
        if torch.distributed.get_rank() == 0:
            print(input_dtensor_spec.shape, local_out_shape)
            print(input_dtensor_spec.placements)
            print(shard_out)

        # The code below doesn't work : it doesn't let me change the propery
        args = op_schema.args_schema
        if spec.shape_argnum is not None:
            op_schema.args_schema = (
                args[: spec.shape_argnum]
                + (tuple(local_out_shape),)
                + args[cast(int, spec.shape_argnum) + 1 :]
            )

        if torch.distributed.get_rank() == 0:
            print(op_schema.args_schema)

        return OutputSharding(
            output_spec=DTensorSpec(
                ndim=len(local_out_shape),
                mesh=input_dtensor_spec.mesh,
                placements=shard_out,
                shape=torch.Size(local_out_shape),
            )
        )


register_prop_rule_map("aten.view.default", Tensor.view)
register_prop_rule_map("aten.unsqueeze.default", torch.unsqueeze)
register_prop_rule_map("aten.expand.default", Tensor.expand)
register_prop_rule_map("aten.permute.default", torch.permute)
register_prop_rule_map("aten.repeat.default", Tensor.repeat)
register_prop_rule_map("aten.transpose.int", torch.transpose)
