import argparse

import torch
from transformers import AutoModel, AutoModelForCausalLM, GenerationConfig



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
