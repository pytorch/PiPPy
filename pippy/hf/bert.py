# Copyright (c) Meta Platforms, Inc. and affiliates
import inspect

import torch
import transformers.utils.fx as fx
from transformers.modeling_utils import ModuleUtilsMixin

import pippy.fx
from pippy import PipelineDriverFillDrain, annotate_split_points, PipeSplitWrapper
from pippy.IR import MultiUseParameterConfig, Pipe
from pippy.microbatch import TensorChunkSpec, CustomReducer


@pippy.fx.wrap
def torch_ones_wrapper(*args, **kwargs):
    return torch.ones(*args, **kwargs)


@pippy.fx.wrap
def torch_create_extended_attention_mask_for_decoder_wrapper(*args, **kwargs):
    return ModuleUtilsMixin.create_extended_attention_mask_for_decoder(*args, **kwargs)


class HFBertTracer(fx.HFTracer):
    def trace(self, root, concrete_args=None):
        graph = super().trace(root, concrete_args)
        # HACK to replace HF's non-resolvable wrappers with the original
        # tensor constructor functions
        # Requires this patch to HF: https://github.com/jamesr66a/transformers/commit/ede9d30f36f12be390692617ef76e2928b1612bd
        for node in graph.nodes:
            if node.op == 'call_function':
                if getattr(node.target, '_orig', None) == torch.ones:
                    node.target = torch_ones_wrapper
                elif getattr(node.target, '_orig', None) == ModuleUtilsMixin.create_extended_attention_mask_for_decoder:
                    node.target = torch_create_extended_attention_mask_for_decoder_wrapper
        return graph


def add_split_points(bert, encoders_per_rank):
    for i in range(0, bert.config.num_hidden_layers // encoders_per_rank):
        annotate_split_points(bert,
                              {f'bert.encoder.layer.{i * encoders_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(bert, {'classifier': PipeSplitWrapper.SplitPoint.BEGINNING})
    return bert.config.num_hidden_layers // encoders_per_rank + 2


def wrap(model, training_args, pp_ranks):
    emb_head = 2  # embeddings + head
    master_emb_head = training_args.exclude_master + emb_head  # master + embeddings + head
    num_of_ranks_for_encoders = (len(pp_ranks) - master_emb_head)
    encoders_per_rank = (model.config.num_hidden_layers + num_of_ranks_for_encoders - 1) // num_of_ranks_for_encoders  # a divider of bert.config.num_hidden_layers: [1, 2, 3, 4, 6, 12]
    # print(f"encoders_per_rank = {encoders_per_rank}")
    number_of_workers = emb_head + model.config.num_hidden_layers // encoders_per_rank  # 3 + a divider of bert.config.num_hidden_layers: [4, 5, 6, 7, 9, 15]
    all_worker_ranks = pp_ranks[training_args.exclude_master:training_args.exclude_master + number_of_workers]
    # print(f"number_of_workers = {number_of_workers}")
    add_split_points(model, encoders_per_rank)

    model.config.problem_type = "single_label_classification"  # "regression", "single_label_classification", or "multi_label_classification"
    input_names = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.TRANSMIT
    output_loss_value_spec = {'loss': True, 'logits': False}
    model_config = model.config
    model = Pipe.from_tracing(model, MULTI_USE_PARAM_CONFIG, tracer=HFBertTracer(), concrete_args=concrete_args,
                              output_loss_value_spec=output_loss_value_spec)
    model.config = model_config

    args_chunk_spec = ()
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'token_type_ids': TensorChunkSpec(0),
                         'labels': TensorChunkSpec(0), 'attention_mask': TensorChunkSpec(0)}
    output_chunk_spec = {'loss': CustomReducer(torch.tensor(0.0), lambda a, b: a + b), 'logits': TensorChunkSpec(0)}
    model = PipelineDriverFillDrain(model, training_args.chunks or len(all_worker_ranks),
                                    args_chunk_spec, kwargs_chunk_spec, output_chunk_spec,
                                    world_size=len(all_worker_ranks),
                                    all_ranks=all_worker_ranks,
                                    _debug_mask_minibatches=False,
                                    _record_mem_dumps=bool(training_args.record_mem_dumps),
                                    checkpoint=bool(training_args.checkpoint))
    model.config = model_config

    model.init_data_parallel(dp_group_size=training_args.dp_group_size)

    return model
