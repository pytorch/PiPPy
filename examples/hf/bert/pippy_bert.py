# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
from functools import reduce

import torch
from transformers import BertLMHeadModel, BertConfig

import pippy
import pippy.fx
from pippy import run_pippy
from pippy.microbatch import sum_reducer, TensorChunkSpec
from pippy.events import EventsContext
from pippy.hf import PiPPyHFTracer
from pippy.visualizer import events_to_json


pippy.fx.Tracer.proxy_buffer_attributes = True


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_submod_sizes(model_pipe):
    total_params = 0
    for i, sm in enumerate(model_pipe.split_gm.children()):
        params = get_number_of_params(sm)
        print(f"submod_{i} {params // 10 ** 6}M params")
        total_params += params
    print(f"total {total_params // 10 ** 6}M params")


def run_master(_, args):
    print("Using schedule:", args.schedule)

    bert = BertLMHeadModel(BertConfig(is_decoder=True))
    print(bert.config)
    print(f"BERT total number of params = {get_number_of_params(bert) // 10 ** 6}M")
    # print(bert)

    chunks = args.chunks or args.world_size
    batches = 1
    bs = 1 * chunks
    seq_length = 16

    device = args.device
    bert.to(device)

    bert_input_dict = {
        'input_ids': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
        'labels': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
        'attention_mask': torch.ones(bs, seq_length, device=device)}
    # bert(**bert_input_dict)

    concrete_args = pippy.create_default_args(bert,
                                              except_keys=bert_input_dict.keys())

    output_chunk_spec = {"loss": sum_reducer,
                         "logits": TensorChunkSpec(0)}

    split_policy = pippy.split_into_equal_size(args.world_size)

    print('Instantiating BERT Pipeline')
    pipe_driver = pippy.compile(
        bert,
        num_ranks=args.world_size,
        num_chunks=chunks,
        schedule=args.schedule,
        split_policy=split_policy,
        tracer=PiPPyHFTracer(),
        checkpoint=bool(args.checkpoint),
        output_chunk_spec=output_chunk_spec,
        concrete_args=concrete_args,
    )
    print_submod_sizes(pipe_driver.pipe)

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print('Running BERT pipeline.')
    pipe_visualized_filename = f"{this_file_name}_visualized.json"
    batches_events_contexts = []
    for i in range(batches):
        pipe_driver(**bert_input_dict)
        batches_events_contexts.append(pipe_driver.retrieve_events())

    all_events_contexts: EventsContext = reduce(lambda c1, c2: EventsContext().update(c1).update(c2),
                                                batches_events_contexts, EventsContext())
    with open(pipe_visualized_filename, "w") as f:
        f.write(events_to_json(all_events_contexts))
    print(f"Saved {pipe_visualized_filename}")
    print('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    parser.add_argument("--chunks", type=int, default=None)
    args = parser.parse_args()

    run_pippy(run_master, args)
