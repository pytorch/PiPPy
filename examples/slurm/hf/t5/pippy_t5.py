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
from transformers import T5ForConditionalGeneration, T5Config
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


def torch_ones_wrapper(*args, **kwargs):
    return torch.ones(*args, **kwargs)


def torch_arange_wrapper(*args, **kwargs):
    return torch.arange(*args, **kwargs)


def torch_full_like_wrapper(*args, **kwargs):
    return torch.full_like(*args, **kwargs)


def torch_create_extended_attention_mask_for_decoder_wrapper(*args, **kwargs):
    return ModuleUtilsMixin.create_extended_attention_mask_for_decoder(*args, **kwargs)


def torch_zeros_wrapper(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


class HFT5Tracer(fx.HFTracer):
    def trace(self, root, concrete_args=None):
        graph = super().trace(root, concrete_args)
        # HACK to replace HF's non-resolvable wrappers with the original
        # tensor constructor functions
        # Requires this patch to HF: https://github.com/jamesr66a/transformers/commit/ede9d30f36f12be390692617ef76e2928b1612bd
        for node in graph.nodes:
            if node.op == 'call_function':
                if getattr(node.target, '_orig', None) == torch.ones:
                    node.target = torch_ones_wrapper
                elif getattr(node.target, '_orig', None) == torch.arange:
                    node.target = torch_arange_wrapper
                elif getattr(node.target, '_orig', None) == torch.full_like:
                    node.target = torch_full_like_wrapper
                elif getattr(node.target, '_orig', None) == ModuleUtilsMixin.create_extended_attention_mask_for_decoder:
                    node.target = torch_create_extended_attention_mask_for_decoder_wrapper
                elif getattr(node.target, '_orig', None) == torch.zeros:
                    node.target = torch_zeros_wrapper
        return graph


def add_split_points(t5, sm_cnt):
    num_encoders = 6  # encoders
    num_lm_head = 1  # lm head
    total_layers = t5.config.num_decoder_layers + num_encoders + num_lm_head
    layers_per_rank = total_layers // sm_cnt
    residual = total_layers % sm_cnt
    split_after_layer = [ (i * layers_per_rank + (i if i < residual else residual)) for i in range(0, sm_cnt) ]
    split_after_decoder = [ x - num_encoders for x in split_after_layer ]
    for i in range(1, sm_cnt):
        annotate_split_points(t5, {f'decoder.block.{split_after_decoder[i]}': PipeSplitWrapper.SplitPoint.BEGINNING})


def resolve_pg_per_stage(pp_rank):
    assert dp_pg_per_pp_rank
    return dp_pg_per_pp_rank[pp_rank]


def run_master(args, pp_ranks):
    torch.manual_seed(42)

    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)

    device = args.device

    t5_config = T5Config.from_pretrained(args.model_config)
    t5_config.use_cache = False  # don't output `past_key_values`
    t5 = T5ForConditionalGeneration(t5_config)
    t5.to(device)  # TODO: Delete this after https://github.com/pytorch/PiPPy/issues/142
    t5.eval()
    print(t5.config)
    print(f"T5 total number of params = {get_number_of_params(t5) // 10 ** 6}M")

    sm_cnt = len(pp_ranks)
    add_split_points(t5, sm_cnt)
    all_worker_ranks = pp_ranks
    chunks = len(all_worker_ranks)
    bs = args.batch_size * chunks
    seq_length = args.seq_length

    print("Using device:", device)

    torch.manual_seed(args.rank)
    inp = torch.empty(bs, seq_length, dtype=torch.long, device=device).random_(t5.config.vocab_size)
    t5_input_dict = {'input_ids': inp, 'decoder_input_ids': inp,
                     'labels': torch.empty(bs, seq_length, dtype=torch.long, device=device).random_(
                         t5.config.vocab_size - 1)}

    # print(t5)

    hf_tracer = HFT5Tracer()

    input_names = t5_input_dict.keys()
    sig = inspect.signature(t5.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    print('Instantiating T5 Pipeline')
    output_loss_value_spec = {'loss': True, 'logits': False,
                              # 'past_key_values': False,
                              'encoder_last_hidden_state': False}
    t5_pipe = Pipe.from_tracing(t5, MULTI_USE_PARAM_CONFIG, tracer=hf_tracer, concrete_args=concrete_args,
                                output_loss_value_spec=output_loss_value_spec)
    split_gm_children = list(t5_pipe.split_gm.children())
    assert sm_cnt == len(
        split_gm_children), f"sm_cnt = {sm_cnt} len(split_gm_children) = {len(split_gm_children)}"
    # t5_pipe.to(device) TODO: Uncomment this after https://github.com/pytorch/PiPPy/issues/142

    # t5_pipe(**t5_input_dict)

    total_params = 0
    for i, sm in enumerate(t5_pipe.split_gm.children()):
        params = get_number_of_params(sm)
        print(f"submod_{i} {params // 10 ** 6}M params")
        total_params += params
    print(f"total {total_params // 10 ** 6}M params")

    args_chunk_spec = ()
    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'decoder_input_ids': TensorChunkSpec(0),
                         'labels': TensorChunkSpec(0)}
    output_chunk_spec = {'loss': CustomReducer(torch.tensor(0.0), lambda a, b: a + b), 'logits': TensorChunkSpec(0),
                         # 'past_key_values': [[TensorChunkSpec(0) for _ in range(36)] for _ in range(4)],
                         'encoder_last_hidden_state': TensorChunkSpec(0)}
    pipe_driver: PipelineDriverBase = schedules[args.schedule](t5_pipe, chunks, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec,
                                                               world_size=len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                               _debug_mask_minibatches=False,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint))

    pipe_driver.init_data_parallel(dp_group_size=args.dp_group_size, dp_pg_cb=resolve_pg_per_stage)

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print('Running T5 pipeline.')
    pipe_visualized_filename = f"{this_file_name}_visualized_{args.rank}.json"
    batches_events_contexts = []
    for i in range(args.batches):
        pipe_driver(**t5_input_dict)
        if args.visualize:
            batches_events_contexts.append(pipe_driver.retrieve_events())

    if args.visualize:
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
    # Bumping to 512 threads to avoid hanging
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=512,
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

    # Init DDP process group
    backend = "nccl" if args.cuda else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=options
    )

    global dp_pg_per_pp_rank
    dp_ranks_per_pp_rank = torch.arange(args.world_size).reshape(args.pp_group_size, args.dp_group_size).tolist()
    dp_pg_per_pp_rank = [torch.distributed.new_group(ranks) for ranks in dp_ranks_per_pp_rank]

    pp_ranks_per_dp_group = [[i * args.dp_group_size + rank for i in range(args.pp_group_size)]
                             for rank in range(args.dp_group_size)]

    global dp_pg_for_reference
    dp_pg_for_reference = torch.distributed.new_group(list(range(args.dp_group_size)))

    if rank >= 0 and rank // args.dp_group_size == 0:
        args.rank = rank
        run_master(args, pp_ranks_per_dp_group[rank])
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 8)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))

    model_config = os.path.dirname(os.path.realpath(__file__)) + "/" + "t5_200m_config.json"
    parser.add_argument('--model_config', type=str, default=model_config)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=16)

    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--visualize', type=int, default=1, choices=[0, 1])
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=1, choices=[0, 1])
    parser.add_argument('--pp_group_size', type=int, default=8)
    args = parser.parse_args()

    assert args.world_size % args.pp_group_size == 0

    args.dp_group_size = args.world_size // args.pp_group_size

    if args.rank == -1:
        mp.spawn(run_worker, args=(args.world_size, args,), nprocs=args.world_size, join=True)
    elif args.rank < args.world_size:
        run_worker(args.rank, args.world_size, args)
    else:
        print("I'm unused, exiting")
