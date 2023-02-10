# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import inspect
import logging
import os
import time
import gc

import torch
import pippy.fx
from pippy import run_pippy
from pippy.IR import MultiUseParameterConfig, Pipe
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.hf import PiPPyHFTracer
from pippy.microbatch import TensorChunkSpec
from pippy import split_on_size_threshold, split_into_equal_size
from transformers import  AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import OPTModel, BloomModel, BloomConfig, BloomForCausalLM
import requests
from transformers import AutoFeatureExtractor, RegNetModel 

import pdb

from accelerate import init_empty_weights
from datasets import load_dataset
import evaluate
from itertools import chain

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True
gigabyte_size = 1073741824
megabyte_size = 1048576

def cleanup():
    dist.destroy_process_group()

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

VERBOSE = bool(int(os.environ.get('VERBOSE', False)))

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)

pippy.fx.Tracer.proxy_buffer_attributes = True


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_inputs(args):
    # preparing inputs
    # inp = torch.empty(bs, seq_length, dtype=torch.long, device=device).random_(model.config.vocab_size)
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        use_auth_token=True if args.use_auth_token else None,
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",
            cache_dir=args.cache_dir,
            use_auth_token=True if args.use_auth_token else None,
        )
        raw_datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
            cache_dir=args.cache_dir,
            use_auth_token=True if args.use_auth_token else None,
        )
    
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "revision": args.model_revision,
        "use_auth_token": True if args.use_auth_token else None,
    }

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs)
    elif args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    embedding_size = args.model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        args.model.resize_token_embeddings(len(tokenizer))
    column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    if args.block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = lm_datasets["validation"]
    max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
    eval_dataset = eval_dataset.select(range(max_eval_samples))

    return eval_dataset
    

def compute_metrics(preds, labels):
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    metric = evaluate.load("accuracy")
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)


def test_origin_model(args):
    print("load origin model")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, use_cache=True, low_cpu_mem_usage=False)
    print("load success")
    model.to(torch.bfloat16)
    model.eval()
    #print("Start warming up")
    #for i in range(args.warmup_num_batches):
    #    output = model(**model_input_dict)
    args.model = model
    inp = prepare_inputs(args)
    model_input_dict = {'input_ids': torch.tensor(inp['input_ids'])}
    print('Running model origin inference.')
    with torch.no_grad():
        for i in range(args.num_batches):
            pre_time = time.time()
            output = model(**model_input_dict)
            preds = preprocess_logits_for_metrics(output['logits'])
            #accuracy = compute_metrics(preds, labels)
            #print(accuracy)
            print('Model origin inference is finished, time costs {} seconds'.format(time.time()-pre_time))
            del output
            del preds
            gc.collect()
    """
    print('Running model origin generate.')
    pre_time = time.time()
    for i in range(args.num_batches):
        output = model.generate(**model_input_dict)
    # print(output)
    print('Model origin generate is finished, time costs {} seconds'.format(time.time()-pre_time))
    """

def run_master(pp_ranks, args):

    # logger = setup_logger()

    torch.manual_seed(42)

    MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT
    print(f'REPLICATE config: {args.replicate} -> {MULTI_USE_PARAM_CONFIG}')
    print("Using schedule:", args.schedule)
    
    device = args.device
    model = args.model
    model_config = model.config
    model_config.use_cache = False  # don't output `past_key_values`
    model.eval()
    # print(model.config)
    print(f"model total number of params = {get_number_of_params(model) // 10 ** 6}M")

    number_of_workers = len(pp_ranks) - pippy.utils.exclude_master
    print(f"number_of_workers = {number_of_workers}")

    if args.auto_split == "threshold":
        split_policy = split_on_size_threshold(490 * 1e6)
    elif args.auto_split == "equal_size":
        split_policy = split_into_equal_size(number_of_workers)

    all_worker_ranks = pp_ranks[pippy.utils.exclude_master:pippy.utils.exclude_master + number_of_workers]
    chunks = args.chunks or len(all_worker_ranks)

    print("Using device:", device)

    torch.manual_seed(args.rank)

    input_names = 'input_ids'
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    model_init_start = time.time()
    model_pipe = Pipe.from_tracing(model, MULTI_USE_PARAM_CONFIG, tracer=PiPPyHFTracer(), concrete_args=concrete_args,
                                output_loss_value_spec=None, split_policy=split_policy
                                )
   
    model_pipe.defer_stage_init(args.device)

    torch.distributed.barrier()

    if args.rank!=0:
        return

    inp = prepare_inputs(args)

    model_input_dict = {'input_ids': torch.tensor(inp['input_ids'])}

    labels = torch.tensor(inp['labels'])

    split_gm_children = list(model_pipe.split_gm.children())

    total_params = 0
    for i, sm in enumerate(model_pipe.split_gm.children()):
        params = get_number_of_params(sm)
        print(f"submod_{i} {params // 10 ** 6}M params")
        total_params += params
    print(f"total {total_params // 10 ** 6}M params")

    print(model_pipe.split_gm.graph)

    args_chunk_spec = ()

    kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0)}
    output_chunk_spec = {"logits": TensorChunkSpec(0)}


    pipe_driver: PipelineDriverBase = schedules[args.schedule](model_pipe, chunks, args_chunk_spec, kwargs_chunk_spec,
                                                               output_chunk_spec,
                                                               world_size=len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                                )
    model_init_end = time.time()

    print("Model initialization time")
    print("=========================")
    print("{} seconds".format(model_init_end - model_init_start))

    this_file_name = os.path.splitext(os.path.basename(__file__))[0]

    print("Start warming up")
    #for i in range(args.warmup_num_batches):
    #    with torch.no_grad():
    #        output = pipe_driver(**model_input_dict)
    print('Running model pipeline.')
    #print(model_input_dict)
    with torch.no_grad():
        for i in range(args.num_batches):
            pre_time = time.time()
            output = pipe_driver(**model_input_dict)
            preds = preprocess_logits_for_metrics(output['logits'])
            #accuracy = compute_metrics(preds, labels)
            print('Inference is finished, time costs {} seconds'.format(time.time()-pre_time))
    
    if args.test_origin_model:
        test_origin_model(args, model_input_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(str(os.environ.get("PMI_SIZE", 1))))
    parser.add_argument('--rank', type=int, default=int(str(os.environ.get("PMI_RANK", 0))))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29000'))

    parser.add_argument('--model_name', type=str, default='bigscience/bloom-7b1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument('--num_batches', type=int, default=1)
    parser.add_argument('--warmup_num_batches', type=int, default=5)
    # parser.add_argument('--seq_length', type=int, default=16)
    # parser.add_argument('--avg_seqlen', type=int, default=16)
    # parser.add_argument('--max_seqlen', type=int, default=16)
    # parser.add_argument('--seqlen-stdev', type=int, default=10)

    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--pp_group_size', type=int, default=8)
    parser.add_argument('--exclude_master', type=int, default=0, choices=[0, 1])
    parser.add_argument('--auto_split', type=str, default="equal_size")
    parser.add_argument('--run_type', type=str, default="pipe")
    parser.add_argument('--test_origin_model', type=bool, default=False)

    parser.add_argument('--dataset_name', type=str, default="wikitext")
    parser.add_argument('--dataset_config_name', type=str, default="wikitext-2-raw-v1")
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--validation_split_percentage', type=int, default=5)
    parser.add_argument('--use_auth_token', type=bool, default=False)
    parser.add_argument('--use_fast_tokenizer', type=bool, default=True)
    parser.add_argument('--model_revision', type=str, default="main")
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--max_eval_samples', type=int, default=1)
    parser.add_argument('--preprocessing_num_workers', type=int, default=None)
    parser.add_argument('--overwrite_cache', type=bool, default=False)

    args = parser.parse_args()
    if args.run_type == "origin_model":
        test_origin_model(args)
    elif args.run_type == "pipe":
        assert args.world_size % args.pp_group_size == 0
        args.dp_group_size = args.world_size // args.pp_group_size
        args.gspmd = 1
        print("Start load model")
        config = BloomConfig.from_pretrained(args.model_name)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        print("Model has been loaded successfully")
        args.model = model
        run_pippy(run_master, args)
    else:
        raise ValueError(
            "run_type should be either pipe or origin_model."
        )
