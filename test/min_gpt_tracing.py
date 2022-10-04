# Copyright (c) Meta Platforms, Inc. and affiliates
from pippy.IR import Pipe, annotate_split_points, PipeSplitWrapper
import pippy.fx

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

from minGPT.mingpt.utils import set_seed  # type: ignore
set_seed(42)

import numpy as np
import torch

from torch.utils.data import Dataset

class AdditionDataset(Dataset):
    """
    Returns addition problems of up to some number of digits in the inputs. Recall
    that all GPT cares about are sequences of integers, and completing them according to
    patterns in the data. Therefore, we have to somehow encode addition problems
    as a sequence of integers.
    
    The sum of two n-digit numbers gives a third up to (n+1)-digit number. So our
    encoding will simply be the n-digit first number, n-digit second number, 
    and (n+1)-digit result, all simply concatenated together. Because each addition
    problem is so structured, there is no need to bother the model with encoding
    +, =, or other tokens. Each possible sequence has the same length, and simply
    contains the raw digits of the addition problem.
    
    As a few examples, the 2-digit problems:
    - 85 + 50 = 135 becomes the sequence [8, 5, 5, 0, 1, 3, 5]
    - 6 + 39 = 45 becomes the sequence [0, 6, 3, 9, 0, 4, 5]
    etc.
    
    We will also only train GPT on the final (n+1)-digits because the first
    two n-digits are always assumed to be given. So when we give GPT an exam later,
    we will e.g. feed it the sequence [0, 6, 3, 9], which encodes that we'd like
    to add 6 + 39, and hope that the model completes the integer sequence with [0, 4, 5]
    in 3 sequential steps.
    
    fun exercise: does it help if the result is asked to be produced in reverse order?
    """

    def __init__(self, ndigit, split):
        self.split = split # train/test
        self.ndigit = ndigit
        self.vocab_size = 10 # 10 possible digits 0..9
        # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        self.block_size = ndigit + ndigit + ndigit + 1 - 1
        
        # split up all addition problems into either training data or test data
        num = (10**self.ndigit)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = 10**self.ndigit
        a = idx // nd
        b = idx %  nd
        c = a + b
        render = f'%0{self.ndigit}d%0{self.ndigit}d%0{self.ndigit+1}d' % (a,b,c) # e.g. 03+25=28 becomes "0325028" 
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:self.ndigit*2-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y

ndigit = 2
train_dataset = AdditionDataset(ndigit=ndigit, split='train')
test_dataset = AdditionDataset(ndigit=ndigit, split='test')

from minGPT.mingpt.model import GPT, GPTConfig  # type: ignore

# initialize a baby GPT model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, 
                  n_layer=2, n_head=4, n_embd=128)
model = GPT(mconf)
model.eval()

x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
stock_traced = pippy.fx.symbolic_trace(model, concrete_args={'targets': None})
torch.testing.assert_allclose(stock_traced(x)[0], model(x)[0])

# Specify split points
sp_spec = {
    'blocks.0.mlp.3' : PipeSplitWrapper.SplitPoint.END,
    'blocks.1.mlp.3' : PipeSplitWrapper.SplitPoint.END,
}
annotate_split_points(model, sp_spec)

traced_pipe = Pipe.from_tracing(model, concrete_args={'targets': None})
torch.testing.assert_allclose(traced_pipe(x)[0], model(x)[0])

assert list(dict(traced_pipe.split_gm.named_children()).keys()) == ['submod_0', 'submod_1', 'submod_2']

print(traced_pipe.split_gm.code)
"""
def forward(self, idx, targets_1 = None):
    submod_0 = self.submod_0(idx);  idx = None
    getitem = submod_0[0]
    getitem_1 = submod_0[1];  submod_0 = None
    submod_1 = self.submod_1(getitem, getitem_1);  getitem = getitem_1 = None
    getitem_2 = submod_1[0]
    getitem_3 = submod_1[1];  submod_1 = None
    submod_2 = self.submod_2(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    return (submod_2, None)
"""

print(traced_pipe.replicated_params)
"""
[]
"""
