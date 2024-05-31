# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import unittest

import torch

from pippy._backward import stage_backward


d_hid = 512
batch_size = 256


# MLP as a stage module
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


def main(args=None):
    mod = MLPModule(d_hid)
    x = torch.randn(batch_size, d_hid)
    # As in a pipeline stage, the inputs to this stage requires gradients
    x.requires_grad_(True)
    target = torch.randn(batch_size, d_hid)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Make a copy
    ref_mod = copy.deepcopy(mod)
    ref_x = x.detach().requires_grad_(x.requires_grad)
    ref_target = target.detach()

    # Forward and backward in stage manner
    out = mod(x)
    loss = loss_fn(out, target)
    grad_inputs = stage_backward(
        stage_output=loss,
        output_grads=None,
        input_values=(x,),
    )

    # Run reference
    ref_out = ref_mod(ref_x)
    ref_loss = loss_fn(ref_out, ref_target)
    ref_loss.backward()

    torch.testing.assert_close(grad_inputs[0], ref_x.grad)

    # Every rank checks gradients
    for name, p in mod.named_parameters():
        ref_p = ref_mod.get_parameter(name)
        try:
            torch.testing.assert_close(p.grad, ref_p.grad)
        except AssertionError:
            print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
            raise

    print(f"Gradient test passed")


if __name__ == "__main__":
    main()


class TestStageBackward(unittest.TestCase):
    def test_stage_backward(self):
        main()
