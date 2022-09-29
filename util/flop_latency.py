import logging
import os
import re
import subprocess
import sys
from collections import Counter
from multiprocessing import cpu_count
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import pippy.fx
from pathos.multiprocessing import ProcessingPool as Pool
from torch import nn
from torch_mlir import run_pipeline_with_repro_report
from torch_mlir._mlir_libs._mlir.ir import Context, Module
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

# This unused import is necessary for linalg passes to be registered.
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendInvoker,
)  # pylint: disable=unused-import

from pippy.fx.node import Target
from pippy.fx.passes.shape_prop import ShapeProp

pippy.fx.Tracer.proxy_buffer_attributes = True
sys.setrecursionlimit(2500)

LOWERING_PIPELINE = [
    "func.func(refback-generalize-tensor-pad)",
    # Bufferize.
    "func.func(scf-bufferize)",
    "func.func(tm-tensor-bufferize)",
    "func.func(linalg-init-tensor-to-alloc-tensor)",
    "func.func(linalg-bufferize)",
    "func-bufferize",
    "arith-bufferize",
    "func.func(tensor-bufferize)",
    "func.func(finalizing-bufferize)",
    "refback-munge-calling-conventions",
    "refback-insert-rng-globals",
    # Lower to LLVM
    "func.func(tm-tensor-to-loops)",
    "func.func(refback-munge-memref-copy)",
    "func.func(convert-linalg-to-loops)",
    "func.func(lower-affine)",
    "convert-scf-to-cf",
    "func.func(refback-expand-ops-for-llvm)",
    "func.func(arith-expand)",
    "func.func(convert-math-to-llvm)",
    "convert-linalg-to-llvm",
    "convert-memref-to-llvm",
    "func.func(convert-arith-to-llvm)",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
]

PIPELINE = [
    "torchscript-module-to-torch-backend-pipeline",
    "torch-backend-to-linalg-on-tensors-backend-pipeline",
] + LOWERING_PIPELINE


def script_module_with_annotations(test_module, annotations):
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    frozen_recursivescriptmodule = recursivescriptmodule
    torch._C._jit_pass_inline(frozen_recursivescriptmodule.graph)

    type = frozen_recursivescriptmodule._c._type()
    class_annotator.exportNone(type)
    class_annotator.exportPath(type, ["forward"])
    class_annotator.annotateArgs(type, ["forward"], annotations)
    return frozen_recursivescriptmodule, class_annotator


def compile_fx_module_to_mlir_module(fx_mod, shapes_dtypes):
    ordered_in_shapes = [
        shapes_dtypes[n.name] for n in fx_mod.graph.nodes if n.op == "placeholder"
    ]
    recursivescriptmodule, class_annotator = script_module_with_annotations(
        fx_mod,
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


def get_module_shapes(fx_mod, example_inputs, op_filter=None):
    if op_filter is None:

        class _op_filter:
            def __contains__(self, item):
                return True

        op_filter = _op_filter()

    if isinstance(example_inputs, tuple) and len(example_inputs) == 1:
        example_inputs = example_inputs[0]
    ShapeProp(fx_mod).propagate(example_inputs)
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


def extract_single_node_subgraph(orig_module: nn.Module, node: pippy.fx.Node):
    new_graph = pippy.fx.Graph()
    env: Dict[pippy.fx.Node, pippy.fx.Node] = {}
    for input in node.all_input_nodes:
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node

    new_node = new_graph.node_copy(node, lambda x: env[x])
    new_graph.output(new_node)
    new_graph.lint()
    return pippy.fx.GraphModule(orig_module, new_graph)


class DropoutRemover(pippy.fx.Transformer):
    def call_module(self, target: Target, args, kwargs: Dict[str, Any]) -> Any:
        if isinstance(self.submodules[target], nn.Dropout):
            assert len(args) == 1
            return args[0]
        else:
            return super().call_module(target, args, kwargs)


def compile_model_op_by_op(
    model, example_inputs, tracer=None, return_shapes=False, **tracer_kwargs
):
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
        tracer = pippy.fx.Tracer()

    graph = tracer.trace(model, **tracer_kwargs)
    traced_mod = pippy.fx.GraphModule(model, graph)
    traced_mod = DropoutRemover(traced_mod).transform()
    node_shapes_dtypes = get_module_shapes(traced_mod, *example_inputs)

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
        mod_str = str(mlir_module)
        return mod_str

    with Pool(cpu_count() - 1) as pool:
        node_mlir_modules = pool.map(lambda x: (x[0], body(x[1])), node_sub_gms)

    if return_shapes:
        return list(filter(None, node_mlir_modules)), node_shapes_dtypes
    else:
        return list(filter(None, node_mlir_modules))


MLIR_FP_OP_LATENCIES = {
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

LLVM_FP_OP_LATENCIES = {
    # cheaper
    "fcmp": 1,
    # cheap
    "fadd": 2,
    "fsub": 2,
    "fmin": 2,
    "fmax": 2,
    # expensive
    "fmul": 4,
    # most expensive
    "fdiv": 8,
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
        total_latency += MLIR_FP_OP_LATENCIES.get(op.operation.name, 0)

    ctx = Context()
    ctx.allow_unregistered_dialects = True
    mlir_module = Module.parse(mlir_module_str, ctx)

    traverse_op_region_block_iterators(mlir_module.operation, print_op)
    return total_latency


def count_flop_latency_in_mlir_llvm_module(node_name, mlir_module_str):
    total_latency = 0
    os.makedirs("/tmp/mlir_llvm_modules", exist_ok=True)
    mlir_module_fp = f"/tmp/mlir_llvm_modules/{node_name}.llvm.mlir"
    with open(mlir_module_fp, "w") as mlir_file:
        mlir_file.write(mlir_module_str)

    mlir_translate_path = os.environ["MLIR_TRANSLATE_PATH"]
    llvm_ir_fp = mlir_module_fp.replace(".mlir", "")
    subprocess.run(
        [mlir_translate_path, "--mlir-to-llvmir", mlir_module_fp, "-o", llvm_ir_fp]
    )

    opt_path = os.environ["OPT_PATH"]
    unrolled_llvm_ir_fp = llvm_ir_fp.replace(".llvm", ".unrolled.llvm")
    subprocess.run(
        [
            opt_path,
            "-S",
            "--indvars",
            "--loop-unroll",
            "--unroll-count=1000000000",
            llvm_ir_fp,
            "-o",
            unrolled_llvm_ir_fp,
        ]
    )

    reg = re.compile(f"(fcmp|fadd|fsub|fmin|fmax|fmul|fdiv)")
    with open(unrolled_llvm_ir_fp) as unrolled_llvm_ir:
        res = reg.findall(unrolled_llvm_ir.read())
    op_counts = Counter(res)
    for op, count in op_counts.items():
        total_latency += LLVM_FP_OP_LATENCIES[op] * count

    return node_name, total_latency


def count_flop_latency_in_mlir_modules_using_mlir(
    node_mlir_module_strs: List[Tuple[str, str]]
) -> dict[str, int]:
    with Pool(cpu_count() - 1) as pool:
        latencies = pool.map(
            lambda x: (x[0], count_flop_latency_in_mlir_module(x[1])),
            node_mlir_module_strs,
        )
    return dict(latencies)


def count_flop_latency_in_mlir_modules_using_llvm(
    node_mlir_module_strs: List[Tuple[str, str]]
) -> dict[str, int]:
    with Pool(cpu_count() - 1) as pool:
        latencies = pool.map(
            lambda x: count_flop_latency_in_mlir_llvm_module(*x), node_mlir_module_strs
        )
    return dict(latencies)


def compute_graph_slice_traffic(nodes, node_shapes):
    # Collect incoming edges for every slice of the graph in order to
    # compute ingress and egress data (bytes) further down.
    cumu_inputs = []
    for node in nodes:
        if cumu_inputs:
            cumu_inputs.append(
                set((n.name, node.name) for n in node.all_input_nodes) | cumu_inputs[-1]
            )
        else:
            cumu_inputs.append(set((n.name, node.name) for n in node.all_input_nodes))

    # Collect outgoing edges for every slice of the graph in order to
    # compute ingress and egress data (bytes) further down.
    cumu_outputs = []
    for node in reversed(nodes):
        if cumu_outputs:
            cumu_outputs.append(
                set((node.name, n.name) for n in node.users) | cumu_outputs[-1]
            )
        else:
            cumu_outputs.append(set((node.name, n.name) for n in node.users))
    cumu_outputs = cumu_outputs[::-1]

    ingress_traffic = np.zeros((len(nodes), len(nodes)))
    egress_traffic = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            input_edges = (cumu_inputs[j] - cumu_inputs[i]) | set(
                [(n.name, nodes[i].name) for n in nodes[i].all_input_nodes]
            )
            output_edges = (cumu_outputs[i] - cumu_outputs[j]) | set(
                [(nodes[j].name, n.name) for n in nodes[j].users]
            )
            ingress_edges = input_edges - output_edges
            egress_edges = output_edges - input_edges
            ingress_traffic[i, j] = sum(
                np.prod(node_shapes.get(n, [0])[0]) for n, _ in ingress_edges
            )
            egress_traffic[i, j] = sum(
                np.prod(node_shapes.get(n, [0])[0]) for n, _ in egress_edges
            )

    return ingress_traffic, egress_traffic


def estimate_intra_costs(
    model: torch.nn.Module,
    fx_mod: pippy.fx.GraphModule,
    example_inputs: tuple[torch.Tensor, ...],
    submesh_shapes: tuple[tuple[int, int]],
    tracer: pippy.fx.Tracer,
    tracer_kwargs: dict[str, Any],
    max_n_succ_stages=4096,
    n_autosharding_configs=1,
):

    # Lower op-by-op to MLIR modules (matched to op name).
    logging.info("Compiling model through MLIR")
    node_mlir_modules, node_shapes = compile_model_op_by_op(
        model, example_inputs, tracer=tracer, return_shapes=True, **tracer_kwargs
    )
    # Count up all of the floating point ops and multiply by
    # op latency (TODO: arithmetic intensity model).
    logging.info("Counting FP latencies for torch ops")
    if "OPT_PATH" in os.environ:
        latencies_dict = count_flop_latency_in_mlir_modules_using_llvm(
            node_mlir_modules
        )
    else:
        latencies_dict = count_flop_latency_in_mlir_modules_using_mlir(
            node_mlir_modules
        )

    # Not all nodes in an fx graph correspond to kernels/ops (e.g., inputholder),
    # thus we index over only those that do.
    n_latencies = len(latencies_dict)
    latencies_arr = []
    nodes = []
    for i, node in enumerate(fx_mod.graph.nodes):
        if node.name not in latencies_dict:
            continue
        latencies_arr.append((latencies_dict[node.name], node.name, i))
        nodes.append(node)

    ingress_traffic, egress_traffic = compute_graph_slice_traffic(nodes, node_shapes)

    # intra_costs[i, j, k, l] is the total latency of [node_i, ..., node_j]
    # for the kth submesh choice and lth autosharding choice (with inclusive endpoints).
    # Currently we only have one submesh choice (all of the devices) and one autosharding choices (none).
    n_submesh_choices = len(submesh_shapes)
    intra_costs = np.zeros(
        (n_latencies, n_latencies, n_submesh_choices, n_autosharding_configs)
    )
    # Prefix sums such that jth prefix - ith prefix + op latency = latency from ith to jth (inclusive) op.
    cumu_latencies = np.cumsum([lat for lat, *_, in latencies_arr])
    for i in range(n_latencies):
        for j in range(i, n_latencies):
            lat = cumu_latencies[j] - cumu_latencies[i] + latencies_arr[i][0]
            for k, shape in enumerate(submesh_shapes):
                if lat > 0:
                    intra_costs[i, j, k, :] = (
                        lat + ingress_traffic[i, j] + egress_traffic[i, j]
                    ) * np.prod(
                        shape
                    )  # hack factor in order to penalize using infinite devices

    # TODO: obviously this isn't right.
    max_n_succ_stages = np.full(
        (n_latencies, n_latencies, n_submesh_choices, n_autosharding_configs),
        max_n_succ_stages,
    )

    return (
        intra_costs,
        max_n_succ_stages,
        tuple((node_name, graph_idx) for _, node_name, graph_idx in latencies_arr),
    )
