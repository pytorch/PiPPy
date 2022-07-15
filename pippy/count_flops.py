import torch
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.split_module import split_module
from torch_mlir import run_pipeline_with_repro_report
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

# This unused import is necessary for linalg passes to be registered.
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker # pylint: disable=unused-import

torch.fx.Tracer.proxy_buffer_attributes = True

PIPELINE = [
    "torchscript-module-to-torch-backend-pipeline",
    "torch-backend-to-linalg-on-tensors-backend-pipeline",
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


def compile_ts_module_to_mlir_module(mod, in_shapes):
    recursivescriptmodule, class_annotator = script_module_with_annotations(
        mod, [None] + [(in_shape, torch.float32, True) for in_shape in in_shapes]
    )

    mb = ModuleBuilder()
    mb.import_module(recursivescriptmodule._c, class_annotator)
    run_pipeline_with_repro_report(mb.module, ",".join(PIPELINE), "")

    return mb.module


def make_mod_partition():
    partition_counter = 0

    def _mod_partition(_node):
        nonlocal partition_counter
        partition = partition_counter
        partition_counter += 1
        return partition

    return _mod_partition


def find_op_in_submodule(submodule):
    for node in submodule.graph.nodes:
        if node.op in {"call_function", "call_module"}:
            return node.name
    return None


def compile_model_op_by_op(model, example_input):
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
    tracer = torch.fx.Tracer()
    graph = tracer.trace(model)
    traced = torch.fx.GraphModule(model, graph)
    module_with_submodules = split_module(traced, model, make_mod_partition())

    ShapeProp(module_with_submodules).propagate(example_input)

    mlir_modules = []
    for node in module_with_submodules.graph.nodes:
        if node.op == "call_module":
            submodule = module_with_submodules.get_submodule(node.name)
            ts_traced = torch.jit.script(submodule)
            inp_metas = list(
                filter(
                    None,
                    [
                        inp_node.meta.get("tensor_meta")
                        for inp_node in node.all_input_nodes
                    ],
                )
            )
            if inp_metas:
                inp_shapes = tuple(tuple(inp_meta.shape) for inp_meta in inp_metas)
                try:
                    mlir_module = compile_ts_module_to_mlir_module(
                        ts_traced, inp_shapes
                    )
                    main_op_name = find_op_in_submodule(submodule)
                    assert main_op_name, str(submodule.graph)
                    mlir_modules.append((node.name, main_op_name, mlir_module))
                except Exception as e:
                    # TODO: replace with more specific catch after this lands
                    # https://github.com/llvm/torch-mlir/pull/1064

                    # TODO: this happens for "meta" ops (such as aten::size);
                    # check more robustly.
                    print(f"couldn't compile {ts_traced.graph}")
                except ValueError as e:
                    # TODO: when does this happen?
                    assert (
                        e.args[0]
                        == "Arg annotations should have one entry per function parameter (including self)."
                    )

    return mlir_modules


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


def count_flop_latency_in_mlir_module(mlir_module):
    total_latency = 0

    def print_op(op):
        nonlocal total_latency
        total_latency += FP_OP_LATENCIES.get(op.operation.name, 0)

    traverse_op_region_block_iterators(mlir_module.operation, print_op)
    return total_latency
