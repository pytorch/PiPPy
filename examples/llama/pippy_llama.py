# Minimum effort to run this example:
# $ torchrun --nproc-per-node 2 pippy_llama.py

import argparse
import os

import torch
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoTokenizer

from pippy.IR import Pipe, PipeSplitWrapper, annotate_split_points
from pippy.PipelineStage import PipelineStage


def add_split_points(llama, nranks):
    # Cut model by equal number of layers per rank
    layers_per_rank = (llama.config.num_hidden_layers + nranks - 1) // nranks
    print(f"layers_per_rank = {layers_per_rank}")
    for i in range(1, nranks):
        annotate_split_points(
            llama,
            {f'model.layers.{i * layers_per_rank}': PipeSplitWrapper.SplitPoint.BEGINNING},
        )


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args):
    # Create a blank model
    llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    prompts = (
        "How do you", "I like to", "Can I help", "You have to",
        "The weather is", "I have a", "What is your", "You are a",
    )  # bs = 8
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)

    # Move model to `device` and set to evaluation
    llama.to(args.device)
    llama.eval()
    print(llama)

    # Annotate split points
    add_split_points(llama, args.world_size)

    # Create a pipeline stage from the model
    llama_pipe = Pipe.from_tracing(
        llama,
        num_chunks=args.world_size,
        example_args=(inputs['input_ids'],),
    )

    assert len(list(llama_pipe.split_gm.children())) == args.world_size
    if args.rank == 0:
        for i, sm in enumerate(llama_pipe.split_gm.children()):
            print(f"Pipeline stage {i} {get_number_of_params(sm) // 10 ** 6}M params")

    # Create schedule runtime
    stage = PipelineStage(
        llama_pipe,
        args.rank,
        device=args.device,
    )

    # Run
    output = None
    if args.rank == 0:
        stage(inputs['input_ids'])
    elif args.rank == args.world_size - 1:
        output = stage()
    else:
        stage()

    if output is not None:
        next_token_logits = output[0][:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        print(tokenizer.batch_decode(next_token))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)
