import copy
import unittest
import torch
from IR import trace_training_loop, MicroBatchCode

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(512, 512)
        self.lin2 = torch.nn.Linear(512, 512)
        self.lin3 = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.lin3(x) # To test reuse of modules
        x = self.lin(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x

def train_loop(batch, model):
    out = model(batch)
    loss = torch.sum(out)
    loss.backward()

class TestMicroBatchCode(unittest.TestCase):
    def test_trace_micro_batch_code(self):

        model = MyModel()
        model.train()

        traced_train_loop = trace_training_loop(model, lambda b: train_loop(b, model))

        test_batch = torch.randn(5, 512)

        # Test reference training loop
        optim = torch.optim.SGD(model.parameters(), lr=0.001)
        optim.zero_grad()
        train_loop(test_batch, model)
        ref_grads = [copy.deepcopy(p.grad) for p in model.parameters()]

        # Test traced training loop
        optim.zero_grad()
        traced_train_loop(test_batch)
        test_grads = [copy.deepcopy(p.grad) for p in model.parameters()]

        self.assertEqual(len(test_grads), len(ref_grads))
        for test, ref in zip(test_grads, ref_grads):
            torch.testing.assert_allclose(test, ref)

if __name__ == '__main__':
    unittest.main()