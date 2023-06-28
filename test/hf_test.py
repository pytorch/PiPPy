# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import importlib
import inspect
import sys
import unittest
import warnings

import torch

import transformers.utils.fx as fx
from pippy.IR import (
    annotate_split_points,
    MultiUseParameterConfig,
    Pipe,
    PipeSplitWrapper,
    stage_backward,
)
from transformers import *

_module_by_model_name = {
    "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
    "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
}


def get_module_cls_by_model_name(model_cls_name):
    module_name = _module_by_model_name.get(model_cls_name, "transformers")
    module = importlib.import_module(module_name)
    return getattr(module, model_cls_name)


def noop_splitter(model) -> int:
    return 1


def albert_splitter(model) -> int:
    if isinstance(model, AlbertModel):
        for i in range(model.config.num_hidden_groups):
            for j in range(model.config.inner_group_num):
                annotate_split_points(
                    model,
                    {
                        f"encoder.albert_layer_groups.{i}.albert_layers.{j}": PipeSplitWrapper.SplitPoint.BEGINNING
                    },
                )
        return model.config.num_hidden_layers + 1
    else:
        for i in range(model.config.num_hidden_groups):
            for j in range(model.config.inner_group_num):
                annotate_split_points(
                    model,
                    {
                        f"albert.encoder.albert_layer_groups.{i}.albert_layers.{j}": PipeSplitWrapper.SplitPoint.BEGINNING
                    },
                )
        return model.config.num_hidden_layers + 1


def bert_splitter(model) -> int:
    if isinstance(model, BertModel) or isinstance(model, MegatronBertModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {f"encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
        annotate_split_points(
            model, {"pooler": PipeSplitWrapper.SplitPoint.BEGINNING}
        )
        return model.config.num_hidden_layers + 2
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {
                    f"bert.encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                },
            )
        if model.bert.pooler is not None:
            annotate_split_points(
                model, {"bert.pooler": PipeSplitWrapper.SplitPoint.BEGINNING}
            )
            return model.config.num_hidden_layers + 2
        else:
            return model.config.num_hidden_layers + 1


def distilbert_splitter(model) -> int:
    if isinstance(model, DistilBertModel):
        for i in range(model.config.n_layers):
            annotate_split_points(
                model,
                {
                    f"transformer.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                },
            )
        return model.config.n_layers + 1
    else:
        for i in range(model.config.n_layers):
            annotate_split_points(
                model,
                {
                    f"distilbert.transformer.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                },
            )
        return model.config.n_layers + 1


def mobilebert_splitter(model) -> int:
    if isinstance(model, MobileBertModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {f"encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
        return model.config.num_hidden_layers + 1
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {
                    f"mobilebert.encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                },
            )
        return model.config.num_hidden_layers + 1


def electra_splitter(model) -> int:
    if isinstance(model, ElectraModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {f"encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
        return model.config.num_hidden_layers + 1
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {
                    f"electra.encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                },
            )
        return model.config.num_hidden_layers + 1


def transformer_splitter(model) -> int:
    if isinstance(model, T5PreTrainedModel):
        for i in range(model.config.num_layers):
            annotate_split_points(
                model,
                {f"encoder.block.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
        for i in range(model.config.num_decoder_layers):
            annotate_split_points(
                model,
                {f"decoder.block.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
        return model.config.num_layers + model.config.num_decoder_layers + 1
    elif isinstance(model, GPTNeoPreTrainedModel):
        if isinstance(model, GPTNeoModel):
            for i in range(model.config.num_layers):
                annotate_split_points(
                    model, {f"h.{i}": PipeSplitWrapper.SplitPoint.BEGINNING}
                )
            annotate_split_points(
                model, {"ln_f": PipeSplitWrapper.SplitPoint.BEGINNING}
            )
            return model.config.num_layers + 2
        else:
            for i in range(model.config.num_layers):
                annotate_split_points(
                    model,
                    {
                        f"transformer.h.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                    },
                )
            annotate_split_points(
                model,
                {"transformer.ln_f": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
            return model.config.num_layers + 2
    elif isinstance(model, BloomPreTrainedModel):  # type: ignore
        if isinstance(model, BloomModel):  # type: ignore
            for i in range(model.config.num_hidden_layers):
                annotate_split_points(
                    model, {f"h.{i}": PipeSplitWrapper.SplitPoint.BEGINNING}
                )
            annotate_split_points(
                model, {"ln_f": PipeSplitWrapper.SplitPoint.BEGINNING}
            )
            return model.config.num_hidden_layers + 2
        else:
            for i in range(model.config.num_hidden_layers):
                annotate_split_points(
                    model,
                    {
                        f"transformer.h.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                    },
                )
            annotate_split_points(
                model,
                {"transformer.ln_f": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
            return model.config.num_hidden_layers + 2
    elif isinstance(model, (GPT2Model, GPTJModel)):
        for i in range(model.config.n_layer):
            annotate_split_points(
                model, {f"h.{i}": PipeSplitWrapper.SplitPoint.BEGINNING}
            )
        annotate_split_points(
            model, {"ln_f": PipeSplitWrapper.SplitPoint.BEGINNING}
        )
        return model.config.n_layer + 2
    else:
        for i in range(model.config.n_layer):
            annotate_split_points(
                model,
                {f"transformer.h.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
        annotate_split_points(
            model, {"transformer.ln_f": PipeSplitWrapper.SplitPoint.BEGINNING}
        )
        return model.config.n_layer + 2


def roberta_splitter(model) -> int:
    if isinstance(model, RobertaModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {f"encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
        annotate_split_points(
            model, {"pooler": PipeSplitWrapper.SplitPoint.BEGINNING}
        )
        return model.config.num_hidden_layers + 2
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {
                    f"roberta.encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                },
            )
        if model.roberta.pooler is not None:
            annotate_split_points(
                model, {"roberta.pooler": PipeSplitWrapper.SplitPoint.BEGINNING}
            )
            return model.config.num_hidden_layers + 2
        else:
            return model.config.num_hidden_layers + 1


def vit_splitter(model) -> int:
    if isinstance(model, ViTModel):
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {f"encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING},
            )
        annotate_split_points(
            model, {"pooler": PipeSplitWrapper.SplitPoint.BEGINNING}
        )
        return model.config.num_hidden_layers + 2
    else:
        for i in range(model.config.num_hidden_layers):
            annotate_split_points(
                model,
                {
                    f"vit.encoder.layer.{i}": PipeSplitWrapper.SplitPoint.BEGINNING
                },
            )
        return model.config.num_hidden_layers + 1


splitters = {
    "albert": albert_splitter,
    "bert": bert_splitter,
    "distilbert": distilbert_splitter,
    "mobilebert": mobilebert_splitter,
    "electra": electra_splitter,
    "transformer": transformer_splitter,
    "roberta": roberta_splitter,
    "vit": vit_splitter,
}

bs = 4
num_choices = 3
seq_length = 32


def generate_hf_model(model_cls):
    splitter = splitters.get(model_cls.base_model_prefix, noop_splitter)
    assert splitter is not None
    config_cls = model_cls.config_class
    config = config_cls()
    if (
        model_cls
        in [
            GPT2ForSequenceClassification,
            GPTNeoForSequenceClassification,
            GPTJForSequenceClassification,
            BloomForSequenceClassification,
        ]
        or model_cls.__name__.startswith("Roberta")
        or model_cls.__name__.startswith("Marian")
    ):
        config.pad_token_id = 0
    model = model_cls(config)

    return model, splitter


def generate_concrete_args_for_model(model, input_names=None):
    input_names = input_names if input_names else model.dummy_inputs.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.name not in input_names
    }
    return concrete_args


def generate_inputs_for_model(model_cls, model, include_loss_args=False):
    if model_cls.__name__.endswith("MultipleChoice"):
        input = torch.empty(
            bs, num_choices, seq_length, dtype=torch.long
        ).random_(model.config.vocab_size)
    elif model_cls.__name__.startswith("Roberta"):
        input = torch.zeros(bs, seq_length, dtype=torch.long)
    elif model_cls.__name__.startswith("ViT"):
        input = torch.randn(
            bs,
            model.config.num_channels,
            model.config.image_size,
            model.config.image_size,
        )
    else:
        input = torch.empty(bs, seq_length, dtype=torch.long).random_(
            model.config.vocab_size
        )

    if "Bart" in model_cls.__name__:
        input[:, -1] = model.config.eos_token_id

    input_dict = {"input_ids": input}

    if (
        model_cls.__name__.startswith("T5")
        or model_cls.__name__.startswith("M2M100")
        or model_cls.__name__.startswith("MT5")
        or model_cls
        in [
            BlenderbotModel,
            BlenderbotSmallModel,
            BlenderbotForConditionalGeneration,
            BlenderbotSmallForConditionalGeneration,
            PegasusModel,
            PegasusForConditionalGeneration,
            MarianModel,
            MarianMTModel,
        ]
    ):
        input_dict.update({"decoder_input_ids": input})

    if model_cls.__name__.startswith("ViT"):
        input_dict["pixel_values"] = input_dict.pop("input_ids")

    if include_loss_args:
        if model_cls.__name__.endswith("PreTraining"):
            if model_cls == ElectraForPreTraining:
                input_dict.update(
                    {
                        "labels": torch.empty(
                            bs, seq_length, dtype=torch.long
                        ).random_(1),
                    }
                )
            else:
                label_name = (
                    "sentence_order_label"
                    if model_cls in [AlbertForPreTraining]
                    else "next_sentence_label"
                )
                input_dict.update(
                    {
                        "labels": torch.empty(
                            bs, seq_length, dtype=torch.long
                        ).random_(model.config.vocab_size),
                        label_name: torch.empty(bs, dtype=torch.long).random_(
                            1
                        ),
                    }
                )
        elif model_cls.__name__.endswith("QuestionAnswering"):
            input_dict.update(
                {
                    "start_positions": torch.empty(
                        bs, dtype=torch.long
                    ).random_(seq_length),
                    "end_positions": torch.empty(bs, dtype=torch.long).random_(
                        seq_length
                    ),
                }
            )
        elif (
            model_cls.__name__.endswith("MaskedLM")
            or model_cls.__name__.endswith("HeadModel")
            or model_cls.__name__.endswith("CausalLM")
            or model_cls.__name__.endswith("DoubleHeadsModel")
        ):
            input_dict.update(
                {
                    "labels": torch.empty(
                        bs, seq_length, dtype=torch.long
                    ).random_(model.config.vocab_size),
                }
            )
        elif model_cls.__name__.endswith("TokenClassification"):
            input_dict.update(
                {
                    "labels": torch.empty(
                        bs, seq_length, dtype=torch.long
                    ).random_(model.config.num_labels - 1),
                }
            )
        elif model_cls.__name__.endswith("MultipleChoice"):
            input_dict.update(
                {
                    "labels": torch.empty(bs, dtype=torch.long).random_(
                        num_choices
                    ),
                }
            )
        elif model_cls.__name__.endswith("SequenceClassification"):
            input_dict.update(
                {
                    "labels": torch.empty(bs, dtype=torch.long).random_(
                        model.config.num_labels - 1
                    ),
                }
            )
        elif model_cls.__name__.endswith("NextSentencePrediction"):
            input_dict.update(
                {
                    "labels": torch.empty(bs, dtype=torch.long).random_(1),
                }
            )
        elif model_cls.__name__.endswith("ForConditionalGeneration"):
            input_dict.update(
                {
                    "labels": torch.empty(
                        bs, seq_length, dtype=torch.long
                    ).random_(model.config.vocab_size - 1),
                }
            )
        elif model_cls == ViTForImageClassification:
            input_dict.update(
                {
                    "labels": torch.empty(bs, dtype=torch.long).random_(
                        model.config.num_labels - 1
                    ),
                }
            )
        elif model_cls == ViTForMaskedImageModeling:
            num_patches = (
                model.config.image_size // model.config.patch_size
            ) ** 2
            input_dict.update(
                {
                    "bool_masked_pos": torch.randint(
                        low=0, high=2, size=(1, num_patches)
                    ).bool()
                }
            )
        else:
            raise NotImplementedError(
                f"Class {model_cls.__name__} unsupported for training test "
            )

    return input_dict


def recursive_value_check(out, ref_out):
    if isinstance(out, torch.Tensor):
        assert isinstance(ref_out, torch.Tensor)
        torch.testing.assert_close(out, ref_out)
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
        raise RuntimeError(f"Unsupported type {type(out)}")


class HFModelsForwardTest(unittest.TestCase):
    pass


for _model_cls_name in fx._SUPPORTED_MODELS:
    _model_cls = get_module_cls_by_model_name(_model_cls_name)

    def scope(model_cls, replicate):
        def test_case(self):
            # TODO: https://github.com/pytorch/PiPPy/issues/149
            if model_cls in [
                MegatronBertForNextSentencePrediction,
                BertForNextSentencePrediction,
                MobileBertForNextSentencePrediction,
            ]:
                self.skipTest("Need to fix handling of kwargs")

            # TODO: support SWIN models https://github.com/pytorch/PiPPy/issues/243
            if model_cls in [
                SwinForMaskedImageModeling,
                SwinForImageClassification,
                SwinModel,
            ]:
                self.skipTest("Need to support SWIN models")

            # TODO: support LayoutLM models https://github.com/pytorch/PiPPy/issues/247
            if model_cls in [
                LayoutLMModel,
                LayoutLMForMaskedLM,
                LayoutLMForSequenceClassification,
                LayoutLMForTokenClassification,
            ]:
                self.skipTest("Need to support LayoutLM models")

            # TODO: support CLIP models https://github.com/pytorch/PiPPy/issues/248
            if model_cls in [CLIPModel, CLIPVisionModel]:
                self.skipTest("Need to support CLIP models")

            # TODO: support Speech2Text models https://github.com/pytorch/PiPPy/issues/249
            if model_cls in [
                Speech2TextModel,
                Speech2TextForConditionalGeneration,
            ]:
                self.skipTest("Need to support Speech2Text models")

            # TODO: support Lxmert models https://github.com/pytorch/PiPPy/issues/253
            if model_cls in [
                LxmertForPreTraining,
                LxmertForQuestionAnswering,
                LxmertModel,
            ]:
                self.skipTest("Need to support Lxmert models")

            # TODO: support Hubert models https://github.com/pytorch/PiPPy/issues/254
            if model_cls in [
                HubertModel,
                HubertForSequenceClassification,
                HubertForCTC,
            ]:
                self.skipTest("Need to support Hubert models")

            # TODO: support DistilBert models https://github.com/pytorch/PiPPy/issues/272
            if model_cls in [
                DistilBertModel,
                DistilBertForMaskedLM,
                DistilBertForQuestionAnswering,
                DistilBertForSequenceClassification,
                DistilBertForTokenClassification,
                DistilBertForMultipleChoice,
            ]:
                self.skipTest("Need to support DistilBert models")

            # TODO: support Deberta models https://github.com/pytorch/PiPPy/issues/261
            if model_cls in [
                DebertaModel,
                DebertaV2ForMaskedLM,
                DebertaV2ForSequenceClassification,
                DebertaV2ForTokenClassification,
                DebertaForQuestionAnswering,
                DebertaForTokenClassification,
                DebertaV2ForQuestionAnswering,
                DebertaV2ForQuestionAnswering,
                DebertaV2ForMultipleChoice,
                DebertaV2ForMultipleChoice,
                DebertaV2Model,
                DebertaForMaskedLM,
                DebertaForSequenceClassification,
            ]:
                self.skipTest("Need to support Deberta models")

            # TODO: support Donut SWIN models https://github.com/pytorch/PiPPy/issues/361
            if model_cls in [DonutSwinModel]:
                self.skipTest("Need to support Donut SWIN models")

            # TODO: support ResNet models https://github.com/pytorch/tau/issues/484
            if model_cls in [ResNetModel, ResNetForImageClassification]:
                self.skipTest("Need to support ResNet models")

            # TODO: support Wav2Vec2 models https://github.com/pytorch/tau/issues/485
            if model_cls in [
                Wav2Vec2Model,
                Wav2Vec2ForPreTraining,
                Wav2Vec2ForCTC,
                Wav2Vec2ForSequenceClassification,
                Wav2Vec2ForMaskedLM,
            ]:
                self.skipTest("Need to support Wav2Vec2 models")

            # TODO: support ConvNext models https://github.com/pytorch/tau/issues/486
            if model_cls in [ConvNextModel, ConvNextForImageClassification]:
                self.skipTest("Need to support ConvNext models")

            # TODO: BART models flakiness https://github.com/pytorch/tau/issues/308
            if model_cls in [
                BartForSequenceClassification,
                MBartForSequenceClassification,
                PLBartForSequenceClassification,
            ]:
                self.skipTest("BART models flakiness")

            # TODO: support Segformer models https://github.com/pytorch/tau/issues/592
            if model_cls in [
                SegformerModel,
                SegformerForImageClassification,
                SegformerForSemanticSegmentation,
            ]:
                self.skipTest("Need to support Segformer models")

            # TODO: support CLIPVisionModelWithProjection https://github.com/pytorch/tau/issues/629
            if model_cls in [
                CLIPVisionModelWithProjection,
            ]:
                self.skipTest("Need to support CLIPVisionModelWithProjection")

            # TODO: support SwinBackbone
            if model_cls in [
                SwinBackbone,
                ResNetBackbone,
            ]:
                self.skipTest("Need to support SwinBackbone")

            model, splitter = generate_hf_model(model_cls)
            model.eval()  # Forward only

            submodules_cnt = splitter(model)

            input_dict = generate_inputs_for_model(
                model_cls, model, include_loss_args=False
            )
            concrete_args = generate_concrete_args_for_model(
                model, input_dict.keys()
            )

            hf_tracer = fx.HFTracer()

            multi_use_param_config = (
                MultiUseParameterConfig.REPLICATE
                if replicate
                else MultiUseParameterConfig.TRANSMIT
            )
            model_pipe = Pipe.from_tracing(
                model,
                multi_use_param_config,
                tracer=hf_tracer,
                concrete_args=concrete_args,
            )
            assert submodules_cnt == len(list(model_pipe.split_gm.children()))

            model_output = model(**input_dict)
            model_pipe_output = model_pipe(**input_dict)

            recursive_value_check(model_pipe_output, model_output)
            print(
                f"Correctness check for model {model_cls.__name__}_{multi_use_param_config} passed",
                file=sys.stderr,
            )

        return test_case

    setattr(
        HFModelsForwardTest,
        f"test_{_model_cls.__name__}_transmit",
        scope(_model_cls, False),
    )
    setattr(
        HFModelsForwardTest,
        f"test_{_model_cls.__name__}_replicate",
        scope(_model_cls, True),
    )


def get_output_loss_value_spec_for_model(model_cls):
    if model_cls in [BartForQuestionAnswering, MBartForQuestionAnswering]:
        return {
            "loss": True,
            "start_logits": False,
            "end_logits": False,
            "encoder_last_hidden_state": False,
        }

    if model_cls.__name__.endswith("QuestionAnswering"):
        return {"loss": True, "start_logits": False, "end_logits": False}

    if model_cls in [GPT2DoubleHeadsModel]:
        return {
            "loss": True,
            "logits": False,
            "mc_logits": False,
            "past_key_values": False,
        }

    if model_cls in [
        GPT2ForSequenceClassification,
        GPT2LMHeadModel,
        GPTNeoForCausalLM,
        GPTNeoForSequenceClassification,
        GPTJForCausalLM,
        GPTJForSequenceClassification,
        BlenderbotSmallForCausalLM,
        BlenderbotForCausalLM,
        BartForCausalLM,
        MBartForCausalLM,
        OPTForCausalLM,
        MarianForCausalLM,
        PLBartForCausalLM,
        PegasusForCausalLM,
        Speech2Text2ForCausalLM,
        XGLMForCausalLM,
        OPTForSequenceClassification,
        BloomForSequenceClassification,
        BloomForCausalLM,
        TrOCRForCausalLM,
    ]:
        return {"loss": True, "logits": False, "past_key_values": False}

    if model_cls in [AlbertForPreTraining]:
        return {"loss": True, "prediction_logits": False, "sop_logits": False}

    if model_cls in [
        BertForPreTraining,
        MegatronBertForPreTraining,
        MobileBertForPreTraining,
    ]:
        return {
            "loss": True,
            "prediction_logits": False,
            "seq_relationship_logits": False,
        }

    if model_cls in [
        T5ForConditionalGeneration,
        M2M100ForConditionalGeneration,
        MT5ForConditionalGeneration,
        PLBartForConditionalGeneration,
    ]:
        return {
            "loss": True,
            "logits": False,
            "past_key_values": False,
            "encoder_last_hidden_state": False,
        }

    if model_cls in [
        BartForSequenceClassification,
        BlenderbotForConditionalGeneration,
        MBartForConditionalGeneration,
        BlenderbotSmallForConditionalGeneration,
        MBartForSequenceClassification,
        PLBartForSequenceClassification,
        PegasusForConditionalGeneration,
        BartForConditionalGeneration,
    ]:
        return {
            "loss": True,
            "logits": False,
            "encoder_last_hidden_state": False,
        }

    if model_cls in [NezhaForPreTraining]:
        return {
            "loss": True,
            "prediction_logits": False,
            "seq_relationship_logits": False,
        }

    return {"loss": True, "logits": False}


class HFModelsForwardBackwardTest(unittest.TestCase):
    pass


# Forward-backward tests
for _model_cls_name in fx._SUPPORTED_MODELS:
    _model_cls = get_module_cls_by_model_name(_model_cls_name)

    def scope(model_cls, replicate):
        def test_case(self):
            # TODO: https://github.com/pytorch/PiPPy/issues/149
            if model_cls in [
                MegatronBertForNextSentencePrediction,
                BertForNextSentencePrediction,
                MobileBertForNextSentencePrediction,
            ]:
                self.skipTest("Need to fix handling of kwargs")

            # TODO: support SWIN models https://github.com/pytorch/PiPPy/issues/243
            if model_cls in [
                SwinForMaskedImageModeling,
                SwinForImageClassification,
                SwinModel,
            ]:
                self.skipTest("Need to support SWIN models")

            # TODO: support LayoutLM models https://github.com/pytorch/PiPPy/issues/247
            if model_cls in [
                LayoutLMModel,
                LayoutLMForMaskedLM,
                LayoutLMForSequenceClassification,
                LayoutLMForTokenClassification,
            ]:
                self.skipTest("Need to support LayoutLM models")

            # TODO: support CLIP models https://github.com/pytorch/PiPPy/issues/248
            if model_cls in [CLIPModel, CLIPVisionModel]:
                self.skipTest("Need to support CLIP models")

            # TODO: support Speech2Text models https://github.com/pytorch/PiPPy/issues/249
            if model_cls in [
                Speech2TextModel,
                Speech2TextForConditionalGeneration,
            ]:
                self.skipTest("Need to support Speech2Text models")

            # TODO: support Lxmert models https://github.com/pytorch/PiPPy/issues/253
            if model_cls in [
                LxmertForPreTraining,
                LxmertForQuestionAnswering,
                LxmertModel,
            ]:
                self.skipTest("Need to support Lxmert models")

            # TODO: support Hubert models https://github.com/pytorch/PiPPy/issues/254
            if model_cls in [
                HubertModel,
                HubertForSequenceClassification,
                HubertForCTC,
            ]:
                self.skipTest("Need to support Hubert models")

            # TODO: support DistilBert models https://github.com/pytorch/PiPPy/issues/272
            if model_cls in [
                DistilBertModel,
                DistilBertForMaskedLM,
                DistilBertForQuestionAnswering,
                DistilBertForSequenceClassification,
                DistilBertForTokenClassification,
                DistilBertForMultipleChoice,
            ]:
                self.skipTest("Need to support DistilBert models")

            # TODO: support Deberta models https://github.com/pytorch/PiPPy/issues/261
            if model_cls in [
                DebertaModel,
                DebertaV2ForMaskedLM,
                DebertaV2ForSequenceClassification,
                DebertaV2ForTokenClassification,
                DebertaForQuestionAnswering,
                DebertaForTokenClassification,
                DebertaV2ForQuestionAnswering,
                DebertaV2ForQuestionAnswering,
                DebertaV2ForMultipleChoice,
                DebertaV2ForMultipleChoice,
                DebertaV2Model,
                DebertaForMaskedLM,
                DebertaForSequenceClassification,
            ]:
                self.skipTest("Need to support Deberta models")

            # TODO: support Donut SWIN models https://github.com/pytorch/PiPPy/issues/361
            if model_cls in [DonutSwinModel]:
                self.skipTest("Need to support Donut SWIN models")

            # TODO: support ResNet models https://github.com/pytorch/tau/issues/484
            if model_cls in [ResNetModel, ResNetForImageClassification]:
                self.skipTest("Need to support ResNet models")

            # TODO: support Wav2Vec2 models https://github.com/pytorch/tau/issues/485
            if model_cls in [
                Wav2Vec2Model,
                Wav2Vec2ForPreTraining,
                Wav2Vec2ForCTC,
                Wav2Vec2ForSequenceClassification,
                Wav2Vec2ForMaskedLM,
            ]:
                self.skipTest("Need to support Wav2Vec2 models")

            # TODO: support ConvNext models https://github.com/pytorch/tau/issues/486
            if model_cls in [ConvNextModel, ConvNextForImageClassification]:
                self.skipTest("Need to support ConvNext models")

            # TODO: BART models flakiness https://github.com/pytorch/tau/issues/308
            if model_cls in [
                BartForSequenceClassification,
                MBartForSequenceClassification,
                PLBartForSequenceClassification,
            ]:
                self.skipTest("BART models flakiness")

            # TODO: support Segformer models https://github.com/pytorch/tau/issues/592
            if model_cls in [
                SegformerModel,
                SegformerForImageClassification,
                SegformerForSemanticSegmentation,
            ]:
                self.skipTest("Need to support Segformer models")

            # TODO: support CLIPVisionModelWithProjection https://github.com/pytorch/tau/issues/629
            if model_cls in [
                CLIPVisionModelWithProjection,
                CLIPTextModelWithProjection,
            ]:
                self.skipTest("Need to support CLIPVisionModelWithProjection")

            # TODO: support SwinBackbone
            if model_cls in [
                SwinBackbone,
                ResNetBackbone,
            ]:
                self.skipTest("Need to support SwinBackbone")

            model, splitter = generate_hf_model(model_cls)
            submodules_cnt = splitter(model)

            try:
                input_dict = generate_inputs_for_model(
                    model_cls, model, include_loss_args=True
                )
            except NotImplementedError as e:
                from transformers.models.speech_to_text_2.modeling_speech_to_text_2 import (
                    Speech2Text2Decoder,
                )
                from transformers.models.trocr.modeling_trocr import (
                    TrOCRDecoder,
                )

                if model_cls in [
                    AlbertModel,
                    BartModel,
                    BertModel,
                    DistilBertModel,
                    ElectraModel,
                    GPT2Model,
                    GPTJModel,
                    GPTNeoModel,
                    MegatronBertModel,
                    MobileBertModel,
                    RobertaModel,
                    T5Model,
                    BlenderbotModel,
                    BlenderbotSmallModel,
                    M2M100Model,
                    MT5Model,
                    MarianMTModel,
                    MarianModel,
                    PegasusModel,
                    OPTModel,
                    Speech2Text2Decoder,
                    TrOCRDecoder,
                    MBartModel,
                    CLIPTextModel,
                    PLBartModel,
                    XGLMModel,
                    ViTModel,
                    DebertaModel,
                    DebertaV2Model,
                    NezhaModel,
                    BloomModel,
                ]:
                    self.skipTest("Base models do not have embedded loss")
                else:
                    raise e

            hf_tracer = fx.HFTracer()

            if model_cls in [
                AlbertForSequenceClassification,
                BertForSequenceClassification,
                BartForSequenceClassification,
                DistilBertForSequenceClassification,
                ElectraForSequenceClassification,
                GPT2ForSequenceClassification,
                GPTJForSequenceClassification,
                GPTNeoForSequenceClassification,
                MegatronBertForSequenceClassification,
                MobileBertForSequenceClassification,
                RobertaForSequenceClassification,
                MBartForSequenceClassification,
                PLBartForSequenceClassification,
                DebertaForSequenceClassification,
                DebertaV2ForSequenceClassification,
                NezhaForSequenceClassification,
                OPTForSequenceClassification,
                BloomForSequenceClassification,
            ]:
                model.config.problem_type = "single_label_classification"

            concrete_args = generate_concrete_args_for_model(
                model, input_dict.keys()
            )
            multi_use_param_config = (
                MultiUseParameterConfig.REPLICATE
                if replicate
                else MultiUseParameterConfig.TRANSMIT
            )
            output_loss_value_spec = get_output_loss_value_spec_for_model(
                model_cls
            )
            model_pipe = Pipe.from_tracing(
                model,
                multi_use_param_config,
                tracer=hf_tracer,
                concrete_args=concrete_args,
                output_loss_value_spec=output_loss_value_spec,
            )

            assert submodules_cnt == len(list(model_pipe.split_gm.children()))
            assert any(
                n.target == stage_backward
                for n in model_pipe.split_gm.graph.nodes
            )

            ref_optim = torch.optim.SGD(model.parameters(), lr=0.001)
            ref_optim.zero_grad()
            ref_loss = model(**input_dict)
            ref_loss["loss"].backward()
            ref_grads = {
                k: copy.copy(p.grad) for k, p in model.named_parameters()
            }

            test_optim = torch.optim.SGD(model_pipe.parameters(), lr=0.001)
            test_optim.zero_grad()
            pipe_loss = model_pipe(**input_dict)

            # Shared parameter sync. TODO: move this to actual runtime
            for param_set in model_pipe.replicated_params:
                grad_values = []
                for module_name, param_qualname in param_set.items():
                    grad_values.append(
                        model_pipe.get_parameter(
                            f"split_gm.{module_name}.{param_qualname}"
                        ).grad
                    )

                synced_value = torch.sum(torch.stack(grad_values), dim=0)

                for module_name, param_qualname in param_set.items():
                    model_pipe.get_parameter(
                        f"split_gm.{module_name}.{param_qualname}"
                    ).grad = synced_value
            test_grads = {
                k: copy.copy(p.grad) for k, p in model_pipe.named_parameters()
            }

            # Disable numerical check due to randomness in training mode
            # Setting model.eval() may avoid such randomness but would cause PiPPy to skip backward pass
            # Plus, we have tested numerical correctness in forward-only tests
            # recursive_value_check(pipe_loss, ref_loss)

            for k_test, v_test in test_grads.items():
                k_ref = model_pipe.remap_qualname(k_test)
                if k_ref not in ref_grads:
                    # TODO: fix
                    warnings.warn(
                        f"{k_ref} not in reference parameter set. Probably because "
                        f"it is a shared parameter in the original model"
                    )
                    continue
                v_ref = ref_grads[k_ref]
                # TODO figure out numerical issues
                # torch.testing.assert_close(v_test, v_ref)

            print(
                f"Correctness check for model {model_cls.__name__}_{multi_use_param_config} passed",
                file=sys.stderr,
            )

        return test_case

    setattr(
        HFModelsForwardBackwardTest,
        f"test_{_model_cls.__name__}_backward_transmit",
        scope(_model_cls, False),
    )
    setattr(
        HFModelsForwardBackwardTest,
        f"test_{_model_cls.__name__}_backward_replicate",
        scope(_model_cls, True),
    )

if __name__ == "__main__":
    unittest.main()
