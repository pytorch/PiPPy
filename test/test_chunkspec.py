# Copyright (c) Meta Platforms, Inc. and affiliates
import unittest

import torch

from pippy import ArgsChunkSpec, KwargsChunkSpec, pipe_split, pipeline


d_hid = 512
batch_size = 256
chunks = 4

torch.manual_seed(0)


class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y, z=torch.zeros(batch_size, d_hid)):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = torch.relu(x)
        x = x + z
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        pipe_split()
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin2(x)
        x = torch.relu(x)
        return x


def main(args=None):
    mod = ExampleCode()

    x = torch.randn(batch_size, d_hid)
    y = torch.randn(batch_size, d_hid)
    z = torch.randn(batch_size, d_hid)

    with ArgsChunkSpec((0, 0)), KwargsChunkSpec({"z": 0}):
        pipe = pipeline(
            mod,
            chunks,
            example_args=(x, y),
            example_kwargs={"z": z},
        )

    assert pipe.num_stages == 4

    ref = mod(x, y, z)
    out = pipe(x, y, z)[0]
    torch.testing.assert_close(out, ref)
    print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


if __name__ == "__main__":
    main()


class TestChunkSpec(unittest.TestCase):
    def test_chunk_spec(self):
        main()
