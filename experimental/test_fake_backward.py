import torch
from torch._subclasses.fake_tensor import FakeTensorMode

def debug_fake_backwards():
    torch.randn(1, device='cuda')
    # torch.set_default_device('cuda')
    with FakeTensorMode():
        p = torch.randn(4, 2, requires_grad=True, device='cuda')
        x = torch.randn(8, 4, device='cuda')
        y = torch.mm(x, p).square().sum()
        # y.backward()
        pg, = torch.autograd.grad(y, p)


if __name__ == '__main__':
    debug_fake_backwards()
