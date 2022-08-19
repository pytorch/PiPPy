from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
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


@dataclass
class DimSpec:
    """Specifies how an output dimension maps to an input dimension."""

    pass


# Rules that map each dimension of the output to dimensions of the input tensor
DimMap = Tuple[DimSpec, ...]


@dataclass
class Singleton(DimSpec):
    """Output dimension is a singleton"""

    pass


@dataclass
class InputDim(DimSpec):
    """Output dimension maps directly to an input dimension."""

    input_dim: int


@dataclass
class Broadcast(DimSpec):
    """Output is the broadcast of a singleton input dimension."""

    dim: DimSpec
    dim_size: int

    @classmethod
    def new(cls, dim: DimSpec, dim_size: int) -> DimSpec:
        return Broadcast(dim, dim_size)


@dataclass
class NewDim(DimSpec):
    """This is a new dimension created by the op."""

    size: int

    @classmethod
    def new(cls, size: int) -> DimSpec:
        return Singleton() if size == 1 else NewDim(size)


@dataclass
class Repeat(DimSpec):
    """Output dimension is the input dimension repeated n-times."""

    input_dim: DimSpec
    times: int

    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec:
        if times == 1:
            return dim
        elif isinstance(dim, Singleton):
            # repeating a singleton is the same as broadcasting it
            return Broadcast(dim, times)
        else:
            return Repeat(dim, times)


@dataclass
class Flatten(DimSpec):
    """
    Output dimension is a set of input dimensions flattened, keeping
    right-most adjacent elements adjacent in the output.
    """

    input_dims: Sequence[DimSpec]

    @classmethod
    def new(cls, dims: Sequence[DimSpec]) -> DimSpec:
        if len(dims) == 0:
            # flattening a scalar leads to a singleton
            return Singleton()
        elif len(dims) == 1:
            # flattening a single dimension is no-op
            return dims[0]
        else:
            return Flatten(dims)


@dataclass
class Split(DimSpec):
    """
    This dimension is a member of a decomposition of the input dim.
    Note that input_dim itself could be a Flattened set of input dims.
    """

    input_dim: DimSpec
    group_shape: Shape
    split_id: int

    @classmethod
    def new(
        cls, dim: DimSpec, group_shape: Tuple[int, ...], idx: int
    ) -> DimSpec:
        assert len(group_shape) > 0
        if len(group_shape) == 1:
            # not really a group, just return the input dim back
            assert idx == 0
            return dim
        elif group_shape[idx] == 1:
            return Singleton()
        else:
            # remove singletons from group
            # group_mapping = [(new_index, (shape, old_index)) ...]
            group_mapping = list(
                enumerate((s, i) for i, s in enumerate(group_shape) if s != 1)
            )
            new_group_shape = tuple(m[1][0] for m in group_mapping)
            new_idx = next(filter(lambda x: x[1][1] == idx, group_mapping))[0]
            return Split(dim, new_group_shape, new_idx)


def dim_pad_left(ndim: int, min_dims: int) -> DimMap:
    return (Singleton(),) * max(0, min_dims - ndim) + tuple(
        InputDim(i) for i in range(ndim)
    )


def dim_atleast_3d(ndim: int) -> DimMap:
    if ndim == 0:
        return (Singleton(), Singleton(), Singleton())
    elif ndim == 1:
        return (Singleton(), InputDim(0), Singleton())
    elif ndim == 2:
        return (InputDim(0), InputDim(1), Singleton())
    else:
        return tuple(InputDim(i) for i in range(ndim))


def expand(input_shape: Shape, shape: Shape) -> DimMap:
    """Implements broadcast on multiple dimensions"""
    assert len(shape) >= len(input_shape)

    # 1. create padded input dimensions
    padded_input = dim_pad_left(len(input_shape), len(shape))
    # 2. check that input shapes are compatible
    mapping = []
    for p, desired_s in zip(padded_input, shape):
        if isinstance(p, Singleton):
            actual_s = 1
            assert desired_s >= 0
        else:
            assert isinstance(
                p, InputDim
            ), f"DimSpec not supported in expand: {p}"
            actual_s = input_shape[p.input_dim]
            assert actual_s == 1 or desired_s == -1 or desired_s == actual_s
        mapping.append(
            p
            if desired_s in (1, -1) or desired_s == actual_s
            else Broadcast.new(p, desired_s)
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
        return (Singleton(),)
    elif ndim == 1:
        return (InputDim(0),)
    else:
        return (Flatten.new(tuple(InputDim(i) for i in range(ndim))),)


def normalize_dims(
    dims: Union[int, Tuple[int, ...]], ndim: int
) -> Tuple[int, ...]:
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

    return tuple(InputDim(i) for i in dest)


def dim_repeat(ndim: int, sizes: Shape) -> DimMap:
    sizes = normalize_sizes(sizes)
    assert (
        len(sizes) >= ndim
    ), f"Number of dimensions of repeat dims {sizes} can not be smaller than number of dimensions of tensor {ndim}."
    pad = len(sizes) - ndim
    return tuple(Repeat.new(Singleton(), s) for s in sizes[:pad]) + tuple(
        Repeat.new(InputDim(i), s) for i, s in enumerate(sizes[pad:])
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


def view_groups(from_size: Shape, to_size: Shape) -> DimMap:
    """
    A view or reshape operation can be decomposed into a set of 3 types of smaller operations:
    1) Forward a dimension from input to output
    2) Flatten a set of dimensions into a single dimension
    3) Split one dimension into multiple dimensions

    view_groups identifies these operations and returns, for each output dimension, what
    is operation was performed in the input dimension. For example:

        view_groups([2, 3, 4], [2, 12]) -> (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2)))
        )

    - ouptut dimension 0 maps to input dimension 0
    - output dimension 1 maps to a flattened input dimensions 1 and 2


        view_groups([2, 3], [3, 2]) -> (
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
        )

    - in the above, input is flattened into a single dimension and then split
      into two separate dimensions with different sizes from the input.
    """
    from_nelem = _prod(from_size)
    to_size = infer_size(from_nelem, normalize_sizes(to_size))

    assert from_nelem == _prod(to_size), "Total view shape does not add up"

    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)

    result_pp = []

    while from_idx < from_len or to_idx < to_len:
        from_group_dim, to_group_shape = [], []

        if from_idx >= from_len:
            f = 1
        else:
            f = from_size[from_idx]
            from_group_dim.append(from_idx)
            from_idx += 1

        if to_idx >= to_len:
            t = 1
        else:
            t = to_size[to_idx]
            to_group_shape.append(t)
            to_idx += 1

        # if any of the groups is singleton, great, we need to backtrack though
        if f == 1 and t != 1:
            # produces ([1], [])
            to_idx -= 1
            to_group_shape = []
        elif f != 1 and t == 1:
            # produces ([], [1])
            from_idx -= 1
            from_group_dim = []
        else:
            # produces ([1], [1]),  ([2], [2]), ([2,3], [6])
            while f != t:
                if f < t:
                    nf = from_size[from_idx]
                    from_group_dim.append(from_idx)
                    from_idx += 1
                    f *= nf
                else:
                    nt = to_size[to_idx]
                    to_group_shape.append(nt)
                    to_idx += 1
                    t *= nt

        if len(to_group_shape) > 0:
            flattened = Flatten.new(
                tuple(
                    InputDim(fi) for fi in from_group_dim if from_size[fi] > 1
                )
            )
            result_pp += [
                Split.new(flattened, tuple(to_group_shape), i)
                for i in range(len(to_group_shape))
            ]

    return tuple(result_pp)


def dim_tile(ndim: int, dims: Tuple[int, ...]) -> DimMap:
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    assert dim1 < ndim
    assert dim2 < ndim
    dimmap = list(InputDim(i) for i in range(ndim))
    swapdim = dimmap[dim1]
    dimmap[dim1] = dimmap[dim2]
    dimmap[dim2] = swapdim
    return tuple(dimmap)


def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    dims = tuple(InputDim(i) for i in range(ndim))
    if dim < 0:
        dim += ndim + 1
    return dims[:dim] + (Singleton(),) + dims[dim:]


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
        dim_map=lambda input, dims: tuple(
            InputDim(i) for i in normalize_dims(dims, input.ndim)
        )
    ),
    torch.ravel: Op(dim_map=lambda tensor: dim_flatten(tensor.ndim)),
    Tensor.repeat: Op(
        dim_map=lambda self, *sizes: dim_repeat(self.ndim, sizes)
    ),
    torch.reshape: Op(
        dim_map=lambda input, shape: view_groups(input.shape, shape),
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
        dim_map=lambda input, *shape: view_groups(input.shape, shape),
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

    Sharding propagation follows mapped dimensions:
    - An output dimension that maps directly to an input dimension is sharded equally
    - An output dimension that is a flattened set of input dimensions can only be
      sharded if only the leftmost flattened dimension is sharded.
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
    assert len(in_shard) == len(mesh_sizes)

    sharded_in_dims: Set[int] = set(
        s.dim for s in in_shard if isinstance(s, Shard)
    )

    def get_dim_size(cmd: DimSpec) -> Tuple[int, Optional[InputDim]]:
        if isinstance(cmd, InputDim):
            return (
                local_in_shape[cmd.input_dim],
                cmd if cmd.input_dim in sharded_in_dims else None,
            )
        elif isinstance(cmd, Flatten):
            for dim in cmd.input_dims[1:]:
                assert (
                    not isinstance(dim, InputDim)
                    or dim.input_dim not in sharded_in_dims
                ), "Only the first member of a Flatten dimension group can be sharded"
            dim0 = cmd.input_dims[0]
            return (
                _prod(get_dim_size(a)[0] for a in cmd.input_dims),
                dim0
                if isinstance(dim0, InputDim)
                and dim0.input_dim in sharded_in_dims
                else None,
            )
        elif isinstance(cmd, Split):
            if cmd.split_id:
                # we will shard only on the first dimension of the group
                # so the shape should be the nominal shape of the component
                return cmd.group_shape[cmd.split_id], None
            else:
                dim_size, in_dim = get_dim_size(cmd.input_dim)
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
                out_size = cmd.group_shape[0]
                assert (
                    out_size % submesh_size == 0
                ), f"Resulting dimension size {out_size} is not divisible by its mesh dimension {submesh_size}."
                return out_size // submesh_size, in_dim
        elif isinstance(cmd, Singleton):
            return 1, None
        elif isinstance(cmd, Broadcast):
            return cmd.dim_size, None
        elif isinstance(cmd, NewDim):
            return cmd.size, None
        elif isinstance(cmd, Repeat):
            size, in_dim = get_dim_size(cmd.input_dim)
            assert (
                in_dim is None or in_dim.input_dim not in sharded_in_dims
            ), "Cannot tile sharded dimension."
            return size * cmd.times, None
        else:
            raise RuntimeError(f"cmd not found: {cmd}, in rule: {rule}")

    dim_map = {}
    out_shape = []
    for dim, cmd in enumerate(rule):
        out_size, in_dim = get_dim_size(cmd)
        out_shape.append(out_size)
        if in_dim is not None:
            dim_map[in_dim.input_dim] = dim

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

        # note we are passing _local_ tensor shapes
        input_dtensor_spec = op_schema.args_schema[0]

        assert isinstance(
            input_dtensor_spec, DTensorSpec
        ), "Expected first input to be a DTensorSpec"
        global_in_shape = input_dtensor_spec.shape
        assert global_in_shape is not None, "Shape required."

        local_out_shape, shard_out = propagate_shape_and_sharding(
            input_dtensor_spec.placements,
            local_shape(input_dtensor_spec, torch.distributed.get_rank()),
            rules,
            tuple(input_dtensor_spec.mesh.mesh.shape),
        )
        # We only need the local shape to lower he call into the local op
        args = op_schema.args_schema
        if spec.shape_argnum is not None:
            op_schema.args_schema = (
                args[: spec.shape_argnum]
                + (tuple(local_out_shape),)
                + args[cast(int, spec.shape_argnum) + 1 :]
            )

        # We want to infer the output shape
        global_out_shape, _ = propagate_shape_and_sharding(
            input_dtensor_spec.placements,
            tuple(global_in_shape),
            rules,
            tuple(input_dtensor_spec.mesh.mesh.shape),
        )
        return OutputSharding(
            output_spec=DTensorSpec(
                ndim=len(global_out_shape),
                mesh=input_dtensor_spec.mesh,
                placements=shard_out,
                shape=torch.Size(global_out_shape),
            )
        )


register_prop_rule_map("aten.view.default", Tensor.view)
register_prop_rule_map("aten.view.SymInt", Tensor.view)
register_prop_rule_map("aten.unsqueeze.default", torch.unsqueeze)
register_prop_rule_map("aten.expand.default", Tensor.expand)
register_prop_rule_map("aten.permute.default", torch.permute)
register_prop_rule_map("aten.repeat.default", Tensor.repeat)
register_prop_rule_map("aten.transpose.int", torch.transpose)
