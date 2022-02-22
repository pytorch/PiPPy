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

bert = BertModel(BertConfig())
# print(bert)

for i in range(bert.config.num_hidden_layers):
    annotate_split_points(bert, {f'encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
annotate_split_points(bert, {'pooler': PipeSplitWrapper.SplitPoint.BEGINNING})

input_names = bert.dummy_inputs.keys()
sig = inspect.signature(bert.forward)
concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

hf_tracer = fx.HFTracer()

bert_pipe = Pipe.from_tracing(bert, MultiUseParameterConfig.TRANSMIT, tracer=hf_tracer, concrete_args=concrete_args)
print(bert_pipe)

bert_input = torch.zeros(bs, seq_length, dtype=torch.long).random_(bert.config.vocab_size)
bert_output = bert(bert_input)
bert_pipe_output = bert_pipe(bert_input)
