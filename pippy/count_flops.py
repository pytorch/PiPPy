import re
import sys
from multiprocessing import cpu_count
from typing import Dict, Any, List, Tuple

import torch
from pathos.multiprocessing import ProcessingPool as Pool
from torch import fx, nn
from torch.fx.node import Target
from torch.fx.passes.shape_prop import ShapeProp
from torch_mlir import run_pipeline_with_repro_report
from torch_mlir._mlir_libs._mlir.ir import Context, Module
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

# This unused import is necessary for linalg passes to be registered.
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendInvoker,
)  # pylint: disable=unused-import

torch.fx.Tracer.proxy_buffer_attributes = True
sys.setrecursionlimit(2500)

PIPELINE = [
    "torchscript-module-to-torch-backend-pipeline",
    "torch-backend-to-linalg-on-tensors-backend-pipeline",
    "func.func(linalg-init-tensor-to-alloc-tensor)",
    "func.func(linalg-bufferize)",
    "func.func(convert-linalg-to-affine-loops)",
    # The workhorse of this script - unroll "affine loops" with trip count less than 1000000 (i.e. all of them)
    # completely.
    "func.func(affine-loop-unroll{ unroll-full unroll-full-threshold=1000000 })",
]


def script_module_with_annotations(test_module, annotations):
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    frozen_recursivescriptmodule = recursivescriptmodule
    class_annotator.exportNone(frozen_recursivescriptmodule._c._type())
    class_annotator.exportPath(frozen_recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        frozen_recursivescriptmodule._c._type(), ["forward"], annotations
    )
    return frozen_recursivescriptmodule, class_annotator


def compile_fx_module_to_mlir_module(fx_mod, shapes_dtypes):
    ts_traced = torch.jit.script(fx_mod)
    ordered_in_shapes = [
        shapes_dtypes[n.name] for n in fx_mod.graph.nodes if n.op == "placeholder"
    ]
    assert len(ordered_in_shapes) + 1 == len(
        list(ts_traced.graph.inputs())
    ), f"shapes don't match {ordered_in_shapes} != {list(ts_traced.graph.inputs())}"
    recursivescriptmodule, class_annotator = script_module_with_annotations(
        ts_traced,
        [None] + [(in_shape, dtype, True) for in_shape, dtype in ordered_in_shapes],
    )

    mb = ModuleBuilder()
    mb.import_module(recursivescriptmodule._c, class_annotator)
    run_pipeline_with_repro_report(mb.module, ",".join(PIPELINE), "")

    return mb.module


def find_op_in_submodule(submodule):
    for node in submodule.graph.nodes:
        if node.op in {"call_function", "call_module", "call_method"}:
            return node.name
    return None


def trace_model(model):
    tracer = torch.fx.Tracer()
    graph = tracer.trace(model)
    traced = torch.fx.GraphModule(model, graph)
    return traced


def get_module_shapes(fx_mod, example_input, op_filter=None):
    if op_filter is None:

        class _op_filter:
            def __contains__(self, item):
                return True

        op_filter = _op_filter()

    ShapeProp(fx_mod).propagate(example_input)
    node_shapes = {}

    def maybe_add_shape(node):
        if node.name in node_shapes:
            return
        meta = node.meta
        if meta["type"] != torch.Tensor:
            return
        meta = meta["tensor_meta"]
        node_shapes[node.name] = (
            tuple(meta.shape),
            torch.float32 if meta.dtype == torch.float64 else meta.dtype,
        )

    for node in filter(lambda n: n.op in op_filter, fx_mod.graph.nodes):
        maybe_add_shape(node)
        for inp_node in node.all_input_nodes:
            maybe_add_shape(inp_node)

    return node_shapes


def extract_single_node_subgraph(orig_module: nn.Module, node: fx.Node):
    new_graph = fx.Graph()
    env: Dict[fx.Node, fx.Node] = {}
    for input in node.all_input_nodes:
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node

    new_node = new_graph.node_copy(node, lambda x: env[x])
    new_graph.output(new_node)
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)


class DropoutRemover(torch.fx.Transformer):
    def call_module(self, target: Target, args, kwargs: Dict[str, Any]) -> Any:
        if isinstance(self.submodules[target], nn.Dropout):
            assert len(args) == 1
            return args[0]
        else:
            return super().call_module(target, args, kwargs)


def compile_model_op_by_op(model, example_input, tracer=None):
    """
    High-level this:

    1. Traces an nn.Module to get a fx graph
    2. Splits the fx graph into submodules, one submodule per op
    3. Runs shape propagation on the parent module so that each submodule has shape refined inputs;
       this shape refinement is necessary for lowering through torch-mlir
    4. TS scripts each submodule (using the refined shapes to produce torch-mlir "annotations")
    5. Compiles each TS module using torch-mlir/mlir **with full loop unrolling**
    6. Map the mlir module back to the relevant identifiers (submodule name and aten op name)

    Returns a list of 3 tuples of (submodule name, aten op name, mlir module);  the mlir module
    can then be passed to count_flop_latency_in_mlir_module to get a latency number, in terms of floating point ops.
    """
    if tracer is None:
        tracer = trace_model

    traced_mod = tracer(model)
    traced_mod = DropoutRemover(traced_mod).transform()
    node_shapes_dtypes = get_module_shapes(traced_mod, example_input)

    reg = re.compile(r"size|getattr")

    node_sub_gms = []
    for node in traced_mod.graph.nodes:
        if node.op not in {"call_module", "call_function", "call_method"}:
            continue
        if reg.search(node._pretty_print_target(node.target)):
            continue
        if not all(n.name in node_shapes_dtypes for n in node.all_input_nodes):
            continue

        sub_gm = extract_single_node_subgraph(model, node)
        node_sub_gms.append((node.name, sub_gm))

    def body(sub_gm):
        mlir_module = compile_fx_module_to_mlir_module(sub_gm, node_shapes_dtypes)
        return str(mlir_module)

    with Pool(cpu_count() - 1) as pool:
        node_mlir_modules = pool.map(lambda x: (x[0], body(x[1])), node_sub_gms)

    return list(filter(None, node_mlir_modules))


FP_OP_LATENCIES = {
    # cheaper
    "arith.cmpf": 1,
    # cheap
    "arith.addf": 2,
    "arith.subf": 2,
    "arith.minf": 2,
    "arith.maxf": 2,
    # expensive
    "arith.mulf": 4,
    # most expensive
    "arith.divf": 8,
}


def traverse_op_region_block_iterators(op, handler):
    for i, region in enumerate(op.regions):
        for j, block in enumerate(region):
            for k, child_op in enumerate(block):
                handler(child_op)
                traverse_op_region_block_iterators(child_op, handler)


def count_flop_latency_in_mlir_module(mlir_module_str):
    total_latency = 0

    def print_op(op):
        nonlocal total_latency
        total_latency += FP_OP_LATENCIES.get(op.operation.name, 0)

    ctx = Context()
    ctx.allow_unregistered_dialects = True
    mlir_module = Module.parse(mlir_module_str, ctx)

    traverse_op_region_block_iterators(mlir_module.operation, print_op)
    return total_latency


def count_flop_latency_in_mlir_modules(node_mlir_module_strs: List[Tuple[str, str]]):
    with Pool(cpu_count() - 1) as pool:
        latencies = pool.map(
            lambda x: (x[0], count_flop_latency_in_mlir_module(x[1])),
            node_mlir_module_strs,
        )
    return dict(latencies)
