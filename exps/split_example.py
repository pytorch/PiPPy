# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

d_hid = 512

class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = torch.mm(x, self.mm_param)
        skip_connection = x
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param)
        x = self.lin(x)
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        x = self.lin(x)
        return x

ec = ExampleCode()

input = torch.randn(53, 512)

# Reference output: full batch
ref_out = ec(input)

# Test output: split batch, process separately, cat together

USE_WHOLE_BATCH = False

if USE_WHOLE_BATCH:
    def _calc_microbatch_split_sizes(chunks : int, dim_size : int):
        # TODO: this splits with the last one bigger because i can't
        # figure out the math to make the last one smaller
        chunk_size = dim_size // chunks

        sizes = []
        examples_counted = 0
        for i in range(chunks):
            if i == chunks - 1:
                sizes.append(dim_size - examples_counted)
                examples_counted += (dim_size - examples_counted)
            else:
                sizes.append(chunk_size)
                examples_counted += chunk_size

        assert examples_counted == dim_size
        return sizes

    split_sizes = _calc_microbatch_split_sizes(chunks=10, dim_size=input.shape[0])

    # prefix sum of sizes

    prefix_sums = []
    sum = 0
    for size in split_sizes:
        sum += size
        prefix_sums.append(sum)

    splits = []
    predecessor = 0

    for sum in prefix_sums:
        splits.append((predecessor, sum))
        predecessor = sum

    split_results = []
    for start, finish in splits:
        new_tensor = torch.zeros_like(input)
        new_tensor[start:finish] = input[start:finish]
        result = ec(new_tensor)
        split_results.append(result[start:finish])

    test_out = torch.cat(split_results, dim=0)

    torch.testing.assert_allclose(test_out, ref_out)
else:
    splits = torch.split(input, 10)

    results = [ec(s) for s in splits]
    test_out = torch.cat(results, dim=0)

    torch.testing.assert_allclose(test_out, ref_out)


# # Test epsilon equivalence

"""
AssertionError: Tensor-likes are not close!

Mismatched elements: 89 / 27136 (0.3%)
Greatest absolute difference: 0.0024471282958984375 at index (50, 494) (up to 1e-05 allowed)
Greatest relative difference: 0.010300797414561929 at index (26, 8) (up to 0.0001 allowed)
"""
