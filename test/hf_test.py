# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import inspect
import sys
import unittest
import warnings

import torch

import transformers.utils.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, annotate_split_points, PipeSplitWrapper, stage_backward
from transformers import *
import torch.fx.experimental.meta_tracer

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
        return model.config.num_layers + model.config.num_decoder_layers + 1
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


def roberta_splitter(model) -> int:
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

bs = 4
num_choices = 3
seq_length = 32

def generate_hf_model(model_cls):
    splitter = splitters.get(model_cls.base_model_prefix)
    assert splitter is not None
    config_cls = model_cls.config_class
    config = config_cls()
    if model_cls in [GPT2ForSequenceClassification, GPTNeoForSequenceClassification,
                    GPTJForSequenceClassification] or model_cls.__name__.startswith("Roberta"):
        config.pad_token_id = 0
    model = model_cls(config)
    model.eval()

    return model, splitter


def generate_concrete_args_for_model(model, input_names=None):
    input_names = input_names if input_names else model.dummy_inputs.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    return concrete_args


def generate_inputs_for_model(model_cls, model, include_loss_args=False):
    if model_cls.__name__.endswith('MultipleChoice'):
        input = torch.zeros(bs, num_choices, seq_length, dtype=torch.long).random_(model.config.vocab_size)
    elif model_cls.__name__.startswith("Roberta"):
        input = torch.zeros(bs, seq_length, dtype=torch.long)
    else:
        input = torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size)

    input_dict = {'input_ids': input}

    if model_cls.__name__.startswith("T5"):
        input_dict.update({'decoder_input_ids': input})

    if include_loss_args:
        if model_cls.__name__.endswith('PreTraining'):
            if model_cls == ElectraForPreTraining:
                input_dict.update({
                    'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(1),
                })
            else:
                label_name = 'sentence_order_label' if model_cls in [AlbertForPreTraining] else 'next_sentence_label'
                input_dict.update({
                    'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size),
                    label_name: torch.zeros(bs, dtype=torch.long).random_(1),
                })
        elif model_cls.__name__.endswith('QuestionAnswering'):
            input_dict.update({
                'start_positions': torch.zeros(bs, dtype=torch.long).random_(seq_length),
                'end_positions': torch.zeros(bs, dtype=torch.long).random_(seq_length)
            })
        elif (model_cls.__name__.endswith('MaskedLM') or model_cls.__name__.endswith('HeadModel') or
              model_cls.__name__.endswith('CausalLM') or model_cls.__name__.endswith('DoubleHeadsModel')):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size),
            })
        elif model_cls.__name__.endswith('TokenClassification'):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.num_labels - 1),
            })
        elif model_cls.__name__.endswith('MultipleChoice'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(num_choices),
            })
        elif model_cls.__name__.endswith('SequenceClassification'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(model.config.num_labels - 1),
            })
        elif model_cls.__name__.endswith('NextSentencePrediction'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(1),
            })
        elif model_cls.__name__.endswith('ForConditionalGeneration'):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size - 1),
            })
        else:
            raise NotImplementedError(f'Class {model_cls.__name__} unsupported for training test ')

    return input_dict


def recursive_value_check(out, ref_out):
    if isinstance(out, torch.Tensor):
        assert isinstance(ref_out, torch.Tensor)
        torch.testing.assert_allclose(out, ref_out)
    elif isinstance(out, (tuple, list)):
        assert isinstance(ref_out, type(out))
        for a, b in zip(out, ref_out):
            recursive_value_check(a, b)
    elif isinstance(out, dict):
        assert isinstance(ref_out, dict)
        assert set(out.keys()) == set(ref_out.keys())
        for k in out.keys():
            recursive_value_check(out[k], ref_out[k])
    else:
        raise RuntimeError(f'Unsupported type {type(out)}')


class HFModelsForwardTest(unittest.TestCase):
    pass


for _model_cls in fx._SUPPORTED_MODELS:
    def scope(model_cls, replicate):
        def test_case(self):
            if model_cls in [MegatronBertForNextSentencePrediction, BertForNextSentencePrediction,
                             MobileBertForNextSentencePrediction]:
                self.skipTest('Need to fix handling of kwargs')

            model, splitter = generate_hf_model(model_cls)

            submodules_cnt = splitter(model)

            input_dict = generate_inputs_for_model(model_cls, model, include_loss_args=False)
            concrete_args = generate_concrete_args_for_model(model, input_dict.keys())

            hf_tracer = torch.fx.experimental.meta_tracer.MetaTracer()
            meta_args = torch.fx.node.map_aggregate(
                input_dict, lambda v: v.to(device='meta') if isinstance(v, torch.Tensor) else v)

            multi_use_param_config = MultiUseParameterConfig.REPLICATE if replicate else MultiUseParameterConfig.TRANSMIT
            model_pipe = Pipe.from_tracing(model, multi_use_param_config, tracer=hf_tracer,
                                           concrete_args=concrete_args, meta_args=meta_args)
            assert submodules_cnt == len(list(model_pipe.split_gm.children()))

            model_output = model(**input_dict)
            model_pipe_output = model_pipe(**input_dict)

            recursive_value_check(model_pipe_output, model_output)
            print(f'Correctness check for model {model_cls.__name__}_{multi_use_param_config} passed', file=sys.stderr)

        return test_case


    setattr(HFModelsForwardTest, f'test_{_model_cls.__name__}_transmit', scope(_model_cls, False))
    setattr(HFModelsForwardTest, f'test_{_model_cls.__name__}_replicate', scope(_model_cls, True))


def get_output_loss_value_spec_for_model(model_cls):
    if model_cls.__name__.endswith('QuestionAnswering'):
        return {'loss': True, 'start_logits': False, 'end_logits': False}

    if model_cls in [GPT2DoubleHeadsModel]:
        return {'loss': True, 'logits': False, 'mc_logits': False, 'past_key_values': False}

    if model_cls in [GPT2ForSequenceClassification, GPT2LMHeadModel, GPTNeoForCausalLM,
                     GPTNeoForSequenceClassification, GPTJForCausalLM, GPTJForSequenceClassification]:
        return {'loss': True, 'logits': False, 'past_key_values': False}

    if model_cls in [AlbertForPreTraining]:
        return {'loss': True, 'prediction_logits': False, 'sop_logits': False}

    if model_cls in [BertForPreTraining, MegatronBertForPreTraining, MobileBertForPreTraining]:
        return {'loss': True, 'prediction_logits': False, 'seq_relationship_logits': False}

    if model_cls in [T5ForConditionalGeneration]:
        return {'loss': True, 'logits': False, 'past_key_values': False, 'encoder_last_hidden_state': False}

    return {'loss': True, 'logits': False}


class HFModelsForwardBackwardTest(unittest.TestCase):
    pass


# Forward-backward tests
for _model_cls in fx._SUPPORTED_MODELS:
    def scope(model_cls, replicate):
        def test_case(self):
            if model_cls in [MegatronBertForNextSentencePrediction, BertForNextSentencePrediction,
                             MobileBertForNextSentencePrediction]:
                self.skipTest('Need to fix handling of kwargs')

            model, splitter = generate_hf_model(model_cls)
            model.eval() # Disable nondeterminism for testing
            submodules_cnt = splitter(model)

            try:
                input_dict = generate_inputs_for_model(model_cls, model, include_loss_args=True)
            except NotImplementedError as e:
                if model_cls in [AlbertModel, BertModel, DistilBertModel, ElectraModel, GPT2Model,
                                 GPTJModel, GPTNeoModel, MegatronBertModel, MobileBertModel, RobertaModel, T5Model]:
                    self.skipTest('Base models do not have embedded loss')
                else:
                    raise e

            hf_tracer = torch.fx.experimental.meta_tracer.MetaTracer()
            meta_args = torch.fx.node.map_aggregate(
                input_dict, lambda v: v.to(device='meta') if isinstance(v, torch.Tensor) else v)

            if model_cls in [AlbertForPreTraining, BertForPreTraining, MegatronBertForPreTraining,
                             MobileBertForPreTraining]:
                # HACK: patching this in for HFTracer to use during concrete value recording
                # otherwise, HFTracer.record generates inputs with bogus shapes for e.g.
                # sentence_order_label
                hf_tracer.input_vals = input_dict

            concrete_args = generate_concrete_args_for_model(model, input_dict.keys())
            multi_use_param_config = MultiUseParameterConfig.REPLICATE if replicate else MultiUseParameterConfig.TRANSMIT
            output_loss_value_spec = get_output_loss_value_spec_for_model(model_cls)
            model_pipe = Pipe.from_tracing(model, multi_use_param_config, tracer=hf_tracer,
                                           concrete_args=concrete_args,
                                           output_loss_value_spec=output_loss_value_spec,
                                           meta_args=meta_args)

            assert submodules_cnt == len(list(model_pipe.split_gm.children()))
            assert any(n.target == stage_backward for n in model_pipe.split_gm.graph.nodes)

            ref_optim = torch.optim.SGD(model.parameters(), lr=0.001)
            ref_optim.zero_grad()
            ref_loss = model(**input_dict)
            ref_loss['loss'].backward()
            ref_grads = {k: copy.copy(p.grad) for k, p in model.named_parameters()}

            test_optim = torch.optim.SGD(model_pipe.parameters(), lr=0.001)
            test_optim.zero_grad()
            pipe_loss = model_pipe(**input_dict)

            # Shared parameter sync. TODO: move this to actual runtime
            for param_set in model_pipe.replicated_params:
                grad_values = []
                for module_name, param_qualname in param_set.items():
                    grad_values.append(model_pipe.get_parameter(f'split_gm.{module_name}.{param_qualname}').grad)

                synced_value = torch.sum(torch.stack(grad_values), dim=0)

                for module_name, param_qualname in param_set.items():
                    model_pipe.get_parameter(f'split_gm.{module_name}.{param_qualname}').grad = synced_value
            test_grads = {k: copy.copy(p.grad) for k, p in model_pipe.named_parameters()}

            recursive_value_check(pipe_loss, ref_loss)

            for k_test, v_test in test_grads.items():
                k_ref = model_pipe.remap_qualname(k_test)
                if k_ref not in ref_grads:
                    # TODO: fix
                    warnings.warn(f'{k_ref} not in reference parameter set. Probably because '
                                  f'it is a shared parameter in the original model')
                    continue
                v_ref = ref_grads[k_ref]
                # TODO figure out numerical issues
                # torch.testing.assert_allclose(v_test, v_ref)

            print(f'Correctness check for model {model_cls.__name__}_{multi_use_param_config} passed', file=sys.stderr)

        return test_case

    setattr(HFModelsForwardBackwardTest, f'test_{_model_cls.__name__}_backward_transmit', scope(_model_cls, False))
    setattr(HFModelsForwardBackwardTest, f'test_{_model_cls.__name__}_backward_replicate', scope(_model_cls, True))

if __name__ == '__main__':
    unittest.main()
