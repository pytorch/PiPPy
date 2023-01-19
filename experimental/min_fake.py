import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode


class MiniModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.randn(4, 2)

    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.p))


def min_forward(m, x):
    return m(x)


if __name__ == '__main__':
    with FakeTensorMode():
        m = MiniModule()
        fake_x = torch.ones(10, 4)

    gm = make_fx(min_forward)(m, fake_x)
    print(gm)
