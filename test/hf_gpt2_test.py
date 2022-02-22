import os
import inspect
import torch
import transformers.utils.fx as fx
from transformers import *

from pippy.IR import MultiUseParameterConfig, Pipe, annotate_split_points, PipeSplitWrapper

REPLICATE = os.environ.get('REPLICATE', '0') != '0'
MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if REPLICATE else MultiUseParameterConfig.TRANSMIT

bs = 4
seq_length = 512

gpt2 = GPT2Model(GPT2Config())
# print(gpt2)

for i in range(gpt2.config.n_layer):
    annotate_split_points(gpt2, {f'h.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
annotate_split_points(gpt2, {'ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})

input_names = gpt2.dummy_inputs.keys()
sig = inspect.signature(gpt2.forward)
concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

hf_tracer = fx.HFTracer()

gpt2_pipe = Pipe.from_tracing(gpt2, MultiUseParameterConfig.TRANSMIT, tracer=hf_tracer, concrete_args=concrete_args)
print(gpt2_pipe)

gpt2_input = torch.zeros(bs, seq_length, dtype=torch.long).random_(gpt2.config.vocab_size)
gpt2_output = gpt2(gpt2_input)
gpt2_pipe_output = gpt2_pipe(gpt2_input)
