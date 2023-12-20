# $ torchrun --nproc-per-node 4 pippy_llama.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pippy import Pipe, PipeSplitWrapper, annotate_split_points, PipelineStage

# Grab the model
llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

prompts = (
    "How do you", "I like to", "Can I help", "You need to",
    "The weather is", "I found a", "What is your", "You are so",
)  # bs = 8
tokenizer.pad_token = tokenizer.eos_token

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
llama.to(device).eval()
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

# Cut model by equal number of layers per rank
layers_per_rank = llama.config.num_hidden_layers // world_size
for i in range(1, world_size):
    annotate_split_points(llama,
        {f"model.layers.{i * layers_per_rank}": PipeSplitWrapper.SplitPoint.BEGINNING})

# Create a pipeline representation from the model
llama_pipe = Pipe.from_tracing(llama, world_size, example_args=(inputs["input_ids"],))

# Create pipeline stage for each rank
torch.distributed.init_process_group(rank=rank, world_size=world_size)
stage = PipelineStage(llama_pipe, rank, device=device)

# Run
if rank == 0:
    args = inputs["input_ids"]
else:
    args = None
output = stage(args)

# Decode
if output is not None:
    next_token_logits = output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(next_token))
