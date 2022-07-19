import inspect
from pprint import pprint

import torch
import transformers.utils.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from transformers import AlbertModel, AlbertConfig

from pippy.count_flops import compile_model_op_by_op, count_flop_latency_in_mlir_modules


class ExampleCode(torch.nn.Module):
    def __init__(self, d_hid=10, bs=4):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)
        self.register_buffer("buffer", torch.randn(bs + 100, d_hid))

    def forward(self, x):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param) + self.buffer[: x.shape[0]]
        x = self.lin(x)
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        x = self.lin(x)
        x = torch.relu(x)
        return x


@torch.fx.wrap
def torch_ones_wrapper(*args, **kwargs):
    return torch.ones(*args, **kwargs)


class HFBertTracer(fx.HFTracer):
    def trace(self, root, concrete_args=None):
        graph = super().trace(root, concrete_args)
        # HACK to replace HF's non-resolvable wrappers with the original
        # tensor constructor functions
        # Requires this patch to HF: https://github.com/jamesr66a/transformers/commit/ede9d30f36f12be390692617ef76e2928b1612bd
        for node in graph.nodes:
            if node.op == "call_function":
                if getattr(node.target, "_orig", None) == torch.ones:
                    node.target = torch_ones_wrapper
        return graph


torch.fx.Tracer.proxy_buffer_attributes = True


def small_test():
    d_hid = 10
    bs = 4
    model = ExampleCode(d_hid, bs)
    x = torch.randn(bs, d_hid)

    node_mlir_module_strs = compile_model_op_by_op(model, x)
    latencies = count_flop_latency_in_mlir_modules(node_mlir_module_strs)
    pprint(latencies)


def albert_test():
    bs = 1
    seq_length = 2

    albert = AlbertModel(AlbertConfig(hidden_size=128, intermediate_size=128))
    albert.eval()

    albert_input = torch.zeros(bs, seq_length, dtype=torch.long).random_(
        albert.config.vocab_size
    )
    input_names = albert.dummy_inputs.keys()
    sig = inspect.signature(albert.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    node_mlir_modules = compile_model_op_by_op(
        albert, albert_input, tracer=HFBertTracer(), concrete_args=concrete_args
    )
    latencies = count_flop_latency_in_mlir_modules(node_mlir_modules)
    pprint(latencies)


if __name__ == "__main__":
    small_test()
    albert_test()
