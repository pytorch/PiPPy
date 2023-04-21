# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import logging
import json
from time import perf_counter
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_input_ids_batch, benchmark, format_to_gb,print_mem_usage,get_number_of_params

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

def run_all(args):
    if os.path.isfile(args.setup_config_path):
            with open(args.setup_config_path) as setup_config_file:
                setup_config = json.load(setup_config_file)
    device =  int(os.getenv("LOCAL_RANK", 0))
    start_time = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name, use_cache=False,
            
        )
    ds_engine = deepspeed.init_inference(model,
                                 mp_size=4,
                                 dtype=torch.half,
                                #  checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                 replace_with_kernel_inject=True)
    model = ds_engine.module
    # model.half()
    model_init_time = (perf_counter()-start_time)*1000
    print("init time is {} ms".format(model_init_time))
    #Generate inputs
    input_ids = generate_input_ids_batch(args.batch_size, args.seq_len)
    input_ids = input_ids.to(device)
    #Run benchmarks
    total_time = benchmark(model,input_ids,args)
    time_per_token = total_time/args.max_tokens
    logger.info("benchmark is done and avegrage time per tokens is %s", time_per_token)

    if os.path.exists(args.log_filename):
        output_file = open(args.log_filename, "a")
    else:
        output_file = open(args.log_filename, "w")

        output_file.write(
            "model_name, batch_size,chunks, seq_len, max_tokens,time_per_token, total_time, model init time\n"
        )

    output_file.write(
                    "{},{},{},{},{},{},{},{}\n".format(
                        args.model_name,
                        args.batch_size,
                        args.chunks,
                        args.seq_len,
                        args.max_tokens,
                        time_per_token,
                        total_time,
                        model_init_time,
                    )
                )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
    parser.add_argument(
        "--rank", type=int, default=int(os.getenv("RANK", -1))
    )
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument(
        "--model_name", type=str, default="bigscience/bloom-560m"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--chunks", type=int, default=1
    )
    parser.add_argument(
        "--cuda", type=int, default=int(torch.cuda.is_available())
    )
    parser.add_argument(
        "--pp_group_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
    parser.add_argument(
        "--seq_len", type=int,default=50,help="",
    )
    parser.add_argument(
        "--batch-size",type=int,default=8,help="",
    )

    parser.add_argument(
        "--max_tokens",type=int,default=10, help="",
    )

    parser.add_argument(
        "--iterations", type=int,default=50,help="",
    )

    parser.add_argument(
        "--log_filename",type=str,default='deepspeed_benchmark_logs.csv', help="",
    )
    parser.add_argument(
        "--setup_config_path",type=str,default='setup_config.json', help="",
    )
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")

    args = parser.parse_args()
    
    run_all(args)
