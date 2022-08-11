# Copyright (c) Meta Platforms, Inc. and affiliates
import inspect

import torch
import transformers.utils.fx as fx

import pippy.fx
from pippy import annotate_split_points, PipeSplitWrapper, Pipe, PipelineDriverFillDrain
from pippy.IR import MultiUseParameterConfig
from pippy.microbatch import TensorChunkSpec, CustomReducer


@pippy.fx.wrap
def torch_arange_wrapper(*args, **kwargs):
    return torch.arange(*args, **kwargs)


#   File "/Users/pbelevich/miniconda3/envs/PiPPy/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py", line 803, in forward
#     position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
# TypeError: arange() received an invalid combination of arguments - got (int, Proxy, device=Attribute, dtype=torch.dtype), but expected one of:
#  * (Number end, *, Tensor out, torch.dtype dtype, torch.layout layout, torch.device device, bool pin_memory, bool requires_grad)
#  * (Number start, Number end, Number step, *, Tensor out, torch.dtype dtype, torch.layout layout, torch.device device, bool pin_memory, bool requires_grad)
class HFGPT2Tracer(fx.HFTracer):
    def trace(self, root, concrete_args=None):
        graph = super().trace(root, concrete_args)
        # HACK to replace HF's non-resolvable wrappers with the original
        # tensor constructor functions
        # Requires this patch to HF: https://github.com/jamesr66a/transformers/commit/ede9d30f36f12be390692617ef76e2928b1612bd
        for node in graph.nodes:
            if node.op == 'call_function':
                if getattr(node.target, '_orig', None) == torch.arange:
                    node.target = torch_arange_wrapper
        return graph


def add_split_points(gpt2, decoders_per_rank):
    for i in range(0, gpt2.config.n_layer // decoders_per_rank):
        annotate_split_points(gpt2, {f'transformer.h.{i * decoders_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING})
    annotate_split_points(gpt2, {f'transformer.ln_f': PipeSplitWrapper.SplitPoint.BEGINNING})
    return gpt2.config.n_layer // decoders_per_rank + 2


def wrap(model, training_args, pp_ranks):
    emb_head = 2  # embeddings + head
    master_emb_head = training_args.exclude_master + emb_head  # master + embeddings + head
    num_of_ranks_for_decoders = (len(pp_ranks) - master_emb_head)
    decoders_per_rank = (model.config.n_layer + num_of_ranks_for_decoders - 1) // num_of_ranks_for_decoders  # a divider of model.config.n_layer: [1, 2, 3, 4, 6, 12]
    # print(f"encoders_per_rank = {decoders_per_rank}")
    number_of_workers = emb_head + model.config.n_layer // decoders_per_rank  # 3 + a divider of model.config.n_layer: [4, 5, 6, 7, 9, 15]
    all_worker_ranks = pp_ranks[training_args.exclude_master:training_args.exclude_master + number_of_workers]
    # print(f"number_of_workers = {decoders_per_rank}")
    add_split_points(model, decoders_per_rank)

    input_names = ['input_ids', 'attention_mask', 'labels']
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.TRANSMIT
    output_loss_value_spec = {'loss': True, 'logits': False, 'past_key_values': False}
    model_config = model.config
    model = Pipe.from_tracing(model, MULTI_USE_PARAM_CONFIG, tracer=HFGPT2Tracer(), concrete_args=concrete_args,
                              output_loss_value_spec=output_loss_value_spec)
    model.config = model_config

    args_chunk_spec = ()
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'labels': TensorChunkSpec(0),
                         'attention_mask': TensorChunkSpec(0)}
    output_chunk_spec = {'loss': CustomReducer(torch.tensor(0.0), lambda a, b: a + b), 'logits': TensorChunkSpec(0),
                         'past_key_values': [[TensorChunkSpec(0) for _ in range(2)] for _ in range(model.config.n_layer)]}
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
