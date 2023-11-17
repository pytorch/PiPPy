# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import unittest

import torch

from pippy.IR import pipe_split, Pipe


d_hid = 512
batch_size = 256

torch.manual_seed(0)


# Basic example
class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = x + y
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param)
        x = self.lin(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin(x)
        x = torch.relu(x)
        return x


# MLP example
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super(MLPModule, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class MultiMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)

    def forward(self, x, y):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        return x - y


def run_worker(args, model_class):
    mod = model_class()
    x = torch.randn(batch_size, d_hid)
    y = torch.randn(batch_size, d_hid)

    pipe = Pipe.from_tracing(
        mod,
        args.chunks,
        example_args=(x, y),
    )

    ref_out = mod(x, y)
    out = pipe(x, y)[0]
    torch.testing.assert_close(out, ref_out)
    print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=int,
        default=4,
    )
    args = parser.parse_args(args)

    for model_class in [ExampleCode, MultiMLP]:
        print("Testing ", model_class.__name__)
        run_worker(args, model_class)


if __name__ == "__main__":
    main()


class TestPipe(unittest.TestCase):
    def test_pipe(self):
        main(args)
