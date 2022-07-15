import torch

from pippy.count_flops import compile_model_op_by_op, count_flop_latency_in_mlir_module


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


def main():
    d_hid = 10
    bs = 4
    model = ExampleCode(d_hid, bs)
    x = torch.randn(bs, d_hid)

    mlir_modules = compile_model_op_by_op(model, x)

    for submod_name, main_op_name, mlir_module in mlir_modules:
        total_latency = count_flop_latency_in_mlir_module(mlir_module)
        print(main_op_name, total_latency)


if __name__ == "__main__":
    main()
