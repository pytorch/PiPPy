# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
import socket
from functools import reduce

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import transformers.utils.fx as fx
from pippy.IR import MultiUseParameterConfig, Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.microbatch import CustomReducer, TensorChunkSpec
from pippy.visualizer import events_to_json
from transformers import BertLMHeadModel, BertConfig
from transformers.modeling_utils import ModuleUtilsMixin

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True


def has_efa() -> bool:
    try:
        import subprocess
        return subprocess.run(["fi_info", "-p", "efa", "-t", "FI_EP_RDM"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        return False


def tp_transports():
    return ["shm", "uv"] if has_efa() else None


schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

torch.fx.Tracer.proxy_buffer_attributes = True


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.fx.wrap
def torch_ones_wrapper(*args, **kwargs):
    return torch.ones(*args, **kwargs)


@torch.fx.wrap
def torch_create_extended_attention_mask_for_decoder_wrapper(*args, **kwargs):
    return ModuleUtilsMixin.create_extended_attention_mask_for_decoder(*args, **kwargs)


class HFBertTracer(fx.HFTracer):
    def trace(self, root, concrete_args=None, method_names=None):
        graph = super().trace(root, concrete_args, method_names)
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
    annotate_split_points(bert, {f'cls': PipeSplitWrapper.SplitPoint.BEGINNING})
    return bert.config.num_hidden_layers // encoders_per_rank + 2


def run_master(args):
    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)

    assert args.world_size >= 4, "This program requires at least 3 workers + 1 master"

    bert = BertLMHeadModel(BertConfig(is_decoder=True))
    bert.eval()
    print(bert.config)
    print(f"BERT total number of params = {get_number_of_params(bert) // 10 ** 6}M")

    emb_head = 2  # embeddings + head
    master_emb_head = 1 + emb_head  # master + embeddings + head
    encoders_per_rank = (bert.config.num_hidden_layers + (args.world_size - master_emb_head) - 1) // (
            args.world_size - master_emb_head)  # a divider of bert.config.num_hidden_layers: [1, 2, 3, 4, 6, 12]
    print(f"encoders_per_rank = {encoders_per_rank}")
    number_of_workers = emb_head + bert.config.num_hidden_layers // encoders_per_rank  # 3 + a divider of bert.config.num_hidden_layers: [4, 5, 6, 7, 9, 15]
    print(f"number_of_workers = {number_of_workers}")

    all_worker_ranks = list(range(1, 1 + number_of_workers))  # exclude master rank = 0
    chunks = len(all_worker_ranks)
    batches = 1
    bs = 1 * chunks
    seq_length = 16

    device = args.device
    print("Using device:", device)

    bert_input_dict = {
        'input_ids': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
        'labels': torch.zeros(bs, seq_length, dtype=torch.long, device=device).random_(bert.config.vocab_size),
        'attention_mask': torch.ones(bs, seq_length, device=device)}

    sm_cnt = add_split_points(bert, encoders_per_rank)
    assert sm_cnt == len(all_worker_ranks), f"sm_cnt = {sm_cnt} all_worker_ranks = {all_worker_ranks}"

    # print(bert)

    hf_tracer = HFBertTracer()

    input_names = bert_input_dict.keys()
    sig = inspect.signature(bert.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    print('Instantiating BERT Pipeline')
    output_loss_value_spec = {'loss': True, 'logits': False}
    bert_pipe = Pipe.from_tracing(bert, MULTI_USE_PARAM_CONFIG, tracer=hf_tracer, concrete_args=concrete_args,
                                  output_loss_value_spec=output_loss_value_spec)
    assert sm_cnt == len(list(bert_pipe.split_gm.children()))
    bert_pipe.to(device)

    # bert_pipe(**bert_input_dict)

    for i, sm in enumerate(bert_pipe.split_gm.children()):
        print(f"submod_{i} {get_number_of_params(sm) // 10 ** 6}M params")

    args_chunk_spec = ()
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'labels': TensorChunkSpec(0),
                         'attention_mask': TensorChunkSpec(0)}
    output_chunk_spec = {'loss': CustomReducer(torch.tensor(0.0), lambda a, b: a + b), 'logits': TensorChunkSpec(0)}
    pipe_driver: PipelineDriverBase = schedules[args.schedule](bert_pipe, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec, len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                               _debug_mask_minibatches=False,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint))

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print('Running BERT pipeline.')
    pipe_visualized_filename = f"{this_file_name}_visualized.json"
    batches_events_contexts = []
    for i in range(batches):
        pipe_driver.run(chunks, **bert_input_dict)
        batches_events_contexts.append(pipe_driver.retrieve_events())

    all_events_contexts: EventsContext = reduce(lambda c1, c2: EventsContext().update(c1).update(c2),
                                                batches_events_contexts, EventsContext())
    with open(pipe_visualized_filename, "w") as f:
        f.write(events_to_json(all_events_contexts))
    print(f"Saved {pipe_visualized_filename}")
    print('Finished')


def run_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # Exclude IB for metadata transport due to lack of EFA support on AWS
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256,
                                              rpc_timeout=1800,
                                              _transports=tp_transports())
    if args.cuda:
        n_devs = torch.cuda.device_count()
        if n_devs > 0:
            dev_id = rank % n_devs
            for i in range(world_size):
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})
        else:
            args.cuda = 0
    args.device = f'cuda:{dev_id}' if args.cuda else 'cpu'
    print(f"rank = {rank} host/pid/device = "
          f"{socket.gethostname()}/{os.getpid()}/{args.device}")
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )
    if rank == 0:
        run_master(args)
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 16)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
