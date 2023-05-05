import argparse

import torch
from transformers import AutoModel, AutoModelForCausalLM, GenerationConfig
gigabyte_size = 1024**3



def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


def print_mem_usage():
    memory_reserved = format_to_gb(torch.cuda.memory_reserved())
    memory_allocated = format_to_gb(torch.cuda.memory_allocated())
    print(
        f"memory_reserved: {memory_reserved} GB, "
        f"memory_allocated: {memory_allocated} GB"
    )


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def generate_input_ids_batch(batch_size, max_sequence_length, vocabulary_size=30522):
    # Create a tensor of shape (batch_size, max_sequence_length) filled with random integers between 0 and vocabulary_size - 1
    input_ids = torch.randint(low=0, high=vocabulary_size, size=(batch_size, max_sequence_length), dtype=torch.long)
    return input_ids



def benchmark(model, input_ids, args):
    with torch.no_grad():
        # Warm up the GPU by running the model once
        _ = model.generate(input_ids, max_new_tokens=args.max_tokens)
        # Measure the elapsed time for multiple iterations of the model
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(args.iterations):
            _ = model.generate(input_ids, max_new_tokens=args.max_tokens)
        end_event.record()
        torch.cuda.synchronize()

    # Calculate the average elapsed time per iteration
    elapsed_time = start_event.elapsed_time(end_event)
    avg_elapsed_time = elapsed_time / args.iterations

    return avg_elapsed_time

