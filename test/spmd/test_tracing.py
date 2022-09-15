# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch.distributed.distributed_c10d import (
    get_global_rank,
    get_world_size,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests
from torch.utils._pytree import tree_flatten, tree_map

import spmd
from spmd.testing.common_utils import (  # type: ignore
    DistTensorTestBase,
    with_comms,
)
from spmd.tensor import (
    DTensor,
    DeviceMesh,
    Replicate,
    Shard,
)
from spmd.tensor.dispatch import operator_dispatch

from functools import partial
from typing import List


class TraceDeviceMeshTestBase:
    def _test_tracing_all_reduce_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                reduced_tensor = mesh.all_reduce(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return reduced_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            reduced_tensor = traced_fn(local_tensor)
            res_num = sum(global_ranks)
            self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)

    def _test_broadcast_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                received_tensor = mesh.broadcast(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            received_tensor = traced_fn(local_tensor)
            res_num = global_ranks[0]
            self.assertEqual(received_tensor, torch.ones(3, 3) * res_num)

    def _test_scatter_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]

            def fn(tensors: List[torch.Tensor]):
                received_tensor = mesh.scatter(tensors, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)([t + 1 for t in scattered_tensors])

            received_tensor = traced_fn(scattered_tensors)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)

    def _test_all_gather_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # each rank have its own tensor, all_gather gives a list
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                gathered_tensors = mesh.all_gather(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return [t * 1 for t in gathered_tensors]

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            gathered_tensors = traced_fn(local_tensor)
            self.assertEqual(len(gathered_tensors), dim_group_size)
            for idx, gathered_tensor in enumerate(gathered_tensors):
                self.assertEqual(
                    gathered_tensor, torch.ones(3, 3) * global_ranks[idx]
                )


class TraceDeviceMesh3DTest(DistTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_broadcast_nd(self):
        self._test_broadcast_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_scatter_nd(self):
        self._test_scatter_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_all_gather_nd(self):
        self._test_all_gather_nd(torch.arange(8).reshape(2, 2, 2))


class TraceDeviceMesh2DTest(DistTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_broadcast_nd(self):
        self._test_broadcast_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_scatter_nd(self):
        self._test_scatter_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_all_gather_nd(self):
        self._test_all_gather_nd(torch.arange(4).reshape(2, 2))


class TraceDistTensorTest(DistTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _test_expand(self, xd, yd, f, mesh, out_spec):
        # trace local graph
        x = xd.redistribute(
            device_mesh=mesh, placements=[Replicate()]
        )._local_tensor
        y = yd.redistribute(
            device_mesh=mesh, placements=[Replicate()]
        )._local_tensor
        traced_f = make_fx(f)(x, y)

        # map intermediate tensors in traced graph to DTensor objects
        node_to_obj = {}

        # map place holder to real input DTensor objects
        def name_to_input(name):
            return xd if "x" in name else yd

        # avoid tracing _op_to_rules and _custom_dispatch_ops as placeholders
        dispatch = partial(
            operator_dispatch,
            op_to_rules=DTensor._op_to_rules,
            custom_dispatch_ops=DTensor._custom_dispatch_ops,
        )

        replacements = {}

        # walk through the traced local graph and expand node with DTensor's
        # dispatch implementation
        for node in traced_f.graph.nodes:
            if node.op == "placeholder":
                node_to_obj[node] = name_to_input(node.name)
            elif isinstance(node.target, torch._ops.OpOverload):
                args = tree_map(lambda n: node_to_obj[n], node.args)
                kwargs = tree_map(lambda n: node_to_obj[n], node.kwargs)

                out = operator_dispatch(
                    node.target,
                    args,
                    kwargs,
                    DTensor._op_to_rules,
                    DTensor._custom_dispatch_ops,
                )
                node_to_obj[node] = out

                # avoid tracing node.target as a placeholder
                dispatch_f = partial(dispatch, node.target)
                # trace DTensor's dispatch logic
                traced_dispatch = make_fx(dispatch_f)(args, kwargs)
                replacements[node] = traced_dispatch
            elif node.op == "output":
                # do nothing, its args will be replaced by dispatcher's
                # output in the next for loop
                pass
            else:
                raise ValueError(f"Unrecognized node {node}")

        # replace enodes in local traced graph with DTensor's dispatch graph
        for node in traced_f.graph.nodes:
            if node not in replacements:
                continue

            # Map DT's dispatch graph input placeholder nodes to the ones in
            # local traced graph. It uses index-based accessing, which is
            # brittle, just for testing purpose.
            flatten_args, _ = tree_flatten(node.args)
            i = 0
            value_remap = {}
            for dtn in traced_dispatch.graph.nodes:
                if dtn.op == "placeholder":
                    value_remap[dtn] = flatten_args[i]
                    i += 1

            # insert DT's dispatch graph to traced local graph.
            with traced_f.graph.inserting_before(node):
                for dtn in traced_dispatch.graph.nodes:
                    if dtn.op == "placeholder":
                        # do nothing, ignore placeholders.
                        pass
                    elif dtn.op == "output":
                        assert (
                            len(dtn.args) == 1 and len(dtn.args[0]) == 1
                        ), f"Expecting single output, but got {dtn.args}"
                        node.replace_all_uses_with(value_remap[dtn.args[0][0]])
                    else:
                        value_remap[dtn] = traced_f.graph.node_copy(
                            dtn, lambda n: value_remap[n]
                        )

        traced_f.graph.lint()
        traced_f.graph.eliminate_dead_code()

        zd = DTensor(
            traced_f(xd._local_tensor, yd._local_tensor), mesh, out_spec
        )
        z = zd.redistribute(
            device_mesh=mesh, placements=[Replicate()]
        ).to_local()

        self.assertEqual(z, f(x, y))

    @with_comms
    def test_simple_expand_replicate_tensor(self):
        def f(x, y):
            return x + y

        mesh = DeviceMesh(self.device_type, torch.arange(2))
        xd = DTensor.from_local(torch.ones(10, 10), mesh, [Replicate()])
        yd = DTensor.from_local(torch.ones(10, 10), mesh, [Replicate()])
        out_spec = [Replicate()]
        self._test_expand(xd, yd, f, mesh, out_spec)

    @with_comms
    def test_simple_expand_shard_tensor(self):
        def f(x, y):
            return x.matmul(y)

        mesh = DeviceMesh(self.device_type, torch.arange(2))

        xd = DTensor.from_local(torch.ones(10, 10), mesh, [Shard(0)])
        yd = DTensor.from_local(torch.ones(10, 10), mesh, [Replicate()])
        out_spec = [Shard(0)]
        self._test_expand(xd, yd, f, mesh, out_spec)


if __name__ == "__main__":
    run_tests()
