import torch
from pippy.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from pippy.fx.passes.operator_support import OperatorSupport
from pippy.fx.passes.tools_common import CALLABLE_NODE_OPS
from pippy.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils._pytree import tree_map

import operator

class CudaGraphsSupport(OperatorSupport):
    # TODO: why is submodules passed here
    def is_node_supported(self, submodules, node: pippy.fx.Node) -> bool:
        if node.op not in CALLABLE_NODE_OPS:
            return False

        if node.target in [torch.ops.aten.embedding_dense_backward.default]:
            return False

        if node.target in [operator.getitem]:
            return True

        found_not_cuda = False

        def meta_fk(meta):
            return meta["val"] if "val" in meta else meta["fake_result"]

        def find_not_cuda(t):
            nonlocal found_not_cuda
            if isinstance(t, torch.Tensor) and t.device.type != 'cuda':
                found_not_cuda = True

        for n in node.all_input_nodes:
            tree_map(find_not_cuda, meta_fk(n.meta))

        tree_map(find_not_cuda, meta_fk(node.meta))

        # NB: factory function is accounted for because the result would be
        # cpu or cuda

        return not found_not_cuda

def partition_cudagraphs(gm, inputs):
    """
    Partition an FX graph into sub-GraphModules that can be validly run under
    CUDA graphs.  For a subgraph to be runnable under CUDA, all of the operations
    must involve CUDA tensors only/
    """

    FakeTensorProp(gm).propagate(*inputs)
    supported_ops = CudaGraphsSupport()
    # TODO: single node partition may be wrong due to the pessimization
    # from copying in and out the data.  Check in benchmarks, perhaps
    partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)
    return fused_graph
