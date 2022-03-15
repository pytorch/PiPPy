import inspect

import torch

import transformers.utils.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, annotate_split_points, PipeSplitWrapper
from transformers import *


def albert_splitter(model) -> int:
    if isinstance(model, AlbertModel):
        for i in range(model.config.num_hidden_groups):
            for j in range(model.config.inner_group_num):
                annotate_split_points(model, {
                    f'encoder.albert_layer_groups.{i}.albert_layers.{j}': PipeSplitWrapper.SplitPoint.BEGINNING
                })
        return model.config.num_hidden_layers + 1
    else:
        for i in range(model.config.num_hidden_groups):
            for j in range(model.config.inner_group_num):
                annotate_split_points(model, {
                    f'albert.encoder.albert_layer_groups.{i}.albert_layers.{j}': PipeSplitWrapper.SplitPoint.BEGINNING
                })
        return model.config.num_hidden_layers + 1


def bert_splitter(model) -> int:
    if isinstance(model, BertModel) or isinstance(model, MegatronBertModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(model, {f'encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        annotate_split_points(model, {'pooler': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.num_hidden_layers + 2
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(model, {f'bert.encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        if model.bert.pooler is not None:
            annotate_split_points(model, {'bert.pooler': PipeSplitWrapper.SplitPoint.BEGINNING})
            return model.config.num_hidden_layers + 2
        else:
            return model.config.num_hidden_layers + 1


def distilbert_splitter(model) -> int:
    if isinstance(model, DistilBertModel):
        for i in range(model.config.n_layers):
            annotate_split_points(model, {f'transformer.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.n_layers + 1
    else:
        for i in range(model.config.n_layers):
            annotate_split_points(model, {f'distilbert.transformer.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.n_layers + 1


def mobilebert_splitter(model) -> int:
    if isinstance(model, MobileBertModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(model, {f'encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.num_hidden_layers + 1
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(model, {f'mobilebert.encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.num_hidden_layers + 1


def electra_splitter(model) -> int:
    if isinstance(model, ElectraModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(model, {f'encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.num_hidden_layers + 1
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(model, {f'electra.encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.num_hidden_layers + 1


def transformer_splitter(model) -> int:
    if isinstance(model, T5PreTrainedModel):
        for i in range(model.config.num_layers):
            annotate_split_points(model, {f'encoder.block.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        for i in range(model.config.num_decoder_layers):
            annotate_split_points(model, {f'decoder.block.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.num_layers + model.config.num_decoder_layers
    elif isinstance(model, GPTNeoPreTrainedModel):
        if isinstance(model, GPTNeoModel):
            for i in range(model.config.num_layers):
                annotate_split_points(model, {f'h.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
            annotate_split_points(model, {'ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})
            return model.config.num_layers + 2
        else:
            for i in range(model.config.num_layers):
                annotate_split_points(model, {f'transformer.h.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
            annotate_split_points(model, {'transformer.ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})
            return model.config.num_layers + 2
    elif isinstance(model, GPT2Model) or isinstance(model, GPTJModel):
        for i in range(model.config.n_layer):
            annotate_split_points(model, {f'h.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        annotate_split_points(model, {'ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.n_layer + 2
    else:
        for i in range(model.config.n_layer):
            annotate_split_points(model, {f'transformer.h.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        annotate_split_points(model, {'transformer.ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.n_layer + 2


def roberta_splitter(model: GPT2PreTrainedModel) -> int:
    if isinstance(model, RobertaModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(model, {f'encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        annotate_split_points(model, {'pooler': PipeSplitWrapper.SplitPoint.BEGINNING})
        return model.config.num_hidden_layers + 2
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(model, {f'roberta.encoder.layer.{i}': PipeSplitWrapper.SplitPoint.BEGINNING})
        if model.roberta.pooler is not None:
            annotate_split_points(model, {'roberta.pooler': PipeSplitWrapper.SplitPoint.BEGINNING})
            return model.config.num_hidden_layers + 2
        else:
            return model.config.num_hidden_layers + 1


splitters = {
    'albert': albert_splitter,
    'bert': bert_splitter,
    'distilbert': distilbert_splitter,
    'mobilebert': mobilebert_splitter,
    'electra': electra_splitter,
    'transformer': transformer_splitter,
    'roberta': roberta_splitter,
}

hf_tracer = fx.HFTracer()

bs = 4
num_choices = 3
seq_length = 32

for model_cls in fx._SUPPORTED_MODELS:
    if model_cls in [T5Model, T5ForConditionalGeneration]:  # https://github.com/jamesr66a/PiPPy/issues/57
        continue
    splitter = splitters.get(model_cls.base_model_prefix)
    print(f"{model_cls.__name__:38} ", end='')
    assert splitter is not None
    config_cls = model_cls.config_class
    config = config_cls()
    if model_cls in [GPT2ForSequenceClassification, GPTNeoForSequenceClassification,
                     GPTJForSequenceClassification] or model_cls.__name__.startswith("Roberta"):
        config.pad_token_id = 0
    model = model_cls(config)
    # print(model)

    submodules_cnt = splitter(model)
    # print(model)

    input_names = model.dummy_inputs.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    model_pipe = Pipe.from_tracing(model, MultiUseParameterConfig.TRANSMIT, tracer=hf_tracer,
                                   concrete_args=concrete_args)
    # print(model_pipe)
    assert submodules_cnt == len(list(model_pipe.split_gm.children()))

    if model_cls.__name__.endswith('MultipleChoice'):
        input = torch.zeros(bs, num_choices, seq_length, dtype=torch.long).random_(model.config.vocab_size)
    elif model_cls.__name__.startswith("Roberta"):
        input = torch.zeros(bs, seq_length, dtype=torch.long)
    else:
        input = torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size)
    model_output = model(input)
    model_pipe_output = model_pipe(input)

    print("OK")
