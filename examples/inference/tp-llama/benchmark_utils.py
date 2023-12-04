import os
import sys
import csv
import torch
import time 
from llama2 import Llama
from generate import text_completion, chat_completion
from typing import List, Literal, Optional, Tuple, TypedDict
import itertools
from torch._dynamo.utils import CompileProfiler
"""
Usage example :

torchrun --nnodes 1 --nproc_per_node 2 benchmark.py  --model_args model_args.json --tokenizer_path /data/home/hamidnazeri/PiPPy/examples/inference/model/models--meta-llama--Llama-2-13b-chat/snapshots/8ebc6a0ac2e4a781c31cb4ad395b1c26c5158c76/tokenizer.model --converted_ckpt_dir converted_checkpoints --batch_size 4 --max_gen_len 50 --num_trials 5 --warmup 2

"""
torch._inductor.config.coordinate_descent_tuning = True
LOCAL_RANK = int(os.environ["LOCAL_RANK"])

def load_model(model_args: str,
               converted_ckpt_dir: str,
               tokenizer_path:str):
    
    """Load the model and move it to the GPU."""
    start_time = time.perf_counter()
    llama_model_and_tok=  Llama.build(
            model_args=model_args,
            converted_ckpt_dir=converted_ckpt_dir,
            tokenizer_path= tokenizer_path,
        )
    model = llama_model_and_tok.model
    tokenizer = llama_model_and_tok.tokenizer
    elapsed_time = time.perf_counter() - start_time
    print(f"Model load time: {elapsed_time} s")
    return model, tokenizer

def compile_model(model, mode="max-autotune"):
    """Compile the model for optimization."""
    start_time = time.perf_counter()
    model= torch.compile(model, mode=mode, fullgraph=True)
    elapsed_time = time.perf_counter() - start_time
    print(f"Compile time: {elapsed_time} s")


def display_memory_stats(step: str):
    """Display peak active and reserved GPU memory."""
    mem_stats = torch.cuda.memory_stats()
    peak_active_gb = mem_stats["active_bytes.all.peak"] / (1024 ** 3)
    peak_reserved_gb = mem_stats["reserved_bytes.all.peak"] / (1024 ** 3)
    print(f"Peak active: {peak_active_gb:.2f} GB | Peak reserved: {peak_reserved_gb:.2f} GB for {step}")
    return peak_active_gb, peak_reserved_gb

def display_model_size(model, model_name: str):
    """Display the number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n---- {model_name} has {total_params / 1e9:.2f} Billion params ----\n")
    return round(total_params / 1e9)

def benchmark_inference_time(pipe, prompt: str, iterations: int) -> float:
    """Benchmark the average inference time."""
    time_profile = []
    cuda_start_event = torch.cuda.Event(enable_timing=True)
    cuda_end_event = torch.cuda.Event(enable_timing=True)
    
    cuda_start_event.record()
    
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        elapsed_time = time.perf_counter() - start_time
        time_profile.append(elapsed_time)
        
    cuda_end_event.record()
    torch.cuda.synchronize()
    
    avg_inference_time_cpu = sum(time_profile) / len(time_profile)
    avg_inference_time_cuda = (cuda_start_event.elapsed_time(cuda_end_event) * 1.0e-3) / iterations
    
    print(f"Average CPU Inference Time: {avg_inference_time_cpu:.4f} s")
    print(f"Average CUDA Inference Time: {avg_inference_time_cuda:.4f} s")
    
    return avg_inference_time_cuda


def benchmark_text_completion(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    logprobs: bool = False,
    echo: bool = False,
    num_trials: int = 1,
    warmup: int=2,
) -> float:
    """
    Benchmark the text_completion function and calculate tokens per second.

    Args:
        model: The language generation model.
        tokenizer: The tokenizer for encoding and decoding text.
        prompts (List[str]): List of text prompts for completion.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
        num_trials (int, optional): Number of trials to run the benchmark. Defaults to 10.

    Returns:
        float: Tokens per second.
    """
    model_size = sum([p.numel() * p.data.element_size() for p in itertools.chain(model.parameters(), model.buffers())])
    total_tokens = 0
    system_time_profiles=[]
    cuda_time_profiles=[]
    cuda_start_event = torch.cuda.Event(enable_timing=True)
    cuda_end_event = torch.cuda.Event(enable_timing=True)
    # with CompileProfiler() as prof:
    #     compiled_model = torch.compile(model, backend="inductor") 
    #     completions = text_completion(
    #         compiled_model,
    #         tokenizer,
    #         prompts,
    #         temperature,
    #         top_p,
    #         max_gen_len,
    #         logprobs,
    #         echo
    #     )
    #     print(prof.report())  
    
    # cuda_start_event.record()
    for _ in range(num_trials):
        start_time = time.perf_counter()
        
        completions = text_completion(
            model,
            tokenizer,
            prompts,
            temperature,
            top_p,
            max_gen_len,
            logprobs,
            echo
        )
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter()-start_time
        system_time_profiles.append(elapsed_time)
    
    # cuda_end_event.record()
    
    for completion in completions:
        decoded_len = len(completion["generation"])
        print(f" len completted texts {len(completions)} and len of each is {decoded_len}")
        print(completion["generation"])
        print("###############################################")
        total_tokens += len(tokenizer.encode(completion["generation"],bos=True, eos=False ))
    
    avg_time = sum(system_time_profiles[warmup:])/ (len(system_time_profiles)-warmup)  
    # avg_time = sum(system_time_profiles)/ len(system_time_profiles) 
    # avg_inference_cuda_time = (cuda_start_event.elapsed_time(cuda_end_event) * 1.0e-3) / num_trials
    tokens_per_second = total_tokens / avg_time
    token_decode_latency= avg_time/total_tokens 
    bandwidth = model_size * tokens_per_second / 1e9
    print(f"total tokens is {total_tokens} and token decoding latency is {token_decode_latency}s per token")
    print(f"Bandwidth achieved: {bandwidth:.02f} GB/s", file=sys.stderr)
    print("=======================================================================")
    if LOCAL_RANK == 0:
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
    print(completions)
    print("****************************************************************")
    return tokens_per_second, bandwidth


def prepare_prompts(prompt: str, tokenizer, max_prompt_len: int, batch_size: int) -> List[str]:
    """
    Replicate and truncate a single prompt based on a specified token length and batch size.

    Args:
        prompt (str): Original text prompt.
        tokenizer: The tokenizer for encoding text.
        max_prompt_len (int): Maximum number of tokens for each replicated prompt.
        batch_size (int): Number of replicated prompts to return.

    Returns:
        List[str]: List of replicated and truncated prompts.
    """
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False)
    print("tokens are encoded in prepare_prompts ")
    total_tokens = len(encoded_prompt)
    truncated_encoded_prompts = encoded_prompt[:max_prompt_len]
    truncated_prompt = tokenizer.decode(truncated_encoded_prompts)

   
    print(f"tottal  token is {total_tokens}, max token is {max_prompt_len} and truncated_encoded_prompts {len(truncated_encoded_prompts)} and itself is {truncated_prompt}")
    print("========================================================================")
    batch_prompts = [truncated_prompt for p in range(batch_size)]
    
    return batch_prompts



def write_to_csv(file_name, values_dict):
    # Check if file exists to decide whether to write headers
    file_exists = os.path.isfile(file_name)
    
    # Open or create the CSV file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file didn't exist
        if not file_exists:
            headers = list(values_dict.keys())
            writer.writerow(headers)
        
        # Write the passed values as a new row
        writer.writerow(list(values_dict.values()))

def read_prompt_from_file(file_name):
    try:
        with open(file_name, 'r') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return None


