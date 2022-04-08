import torch

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(5, 3))
        def hook(x):
            print('hook called')
            return x
        self.param.register_hook(hook)

    def forward(self, x):
        return self.param * torch.relu(x) + self.param


f = Foo()
x = torch.randn(5, 3)

y1 = f(x)
y1_loss = torch.sum(y1)

y2 = f(x)
y2_loss = torch.sum(y2)

def find_accum_grad_fns(init_grad_fn):
    queue = [init_grad_fn]
    seen = {}
    grad_accums = {}

    while len(queue) != 0:
        next_fn = queue.pop(0)
        seen.setdefault(next_fn)

        if isinstance(next_fn, torch._C._functions.AccumulateGrad) and next_fn not in grad_accums:
            grad_accums.setdefault(next_fn)

        for fn, _ in next_fn.next_functions:    
            if fn is None:
                continue
            if fn not in seen:
                queue.append(fn)

    return grad_accums


grad_accums = find_accum_grad_fns(y1_loss.grad_fn)

for accum in grad_accums:
    var = accum.variable
    print(dir(var))

print(grad_accums)

print('***backward 1')
y1_loss.backward()

print('***backward 2')
y2_loss.backward()
