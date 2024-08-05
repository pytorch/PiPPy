# Copyright (c) Meta Platforms, Inc. and affiliates

"""
This script shows how to create llama model in "meta" device mode, partition it
into pipeline stages, and materialize each stage modules from Hugging Face
checkpoints.

Before running the script, please download the following files in the same
directory as this script:
- pytorch_model.bin.index.json
- pytorch_model-00001-of-00002.bin
- pytorch_model-00002-of-00002.bin

Download link:
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main

How to run this script:
$ python meta_init.py

I haven't used a distributed runtime, because I only want to showcase how to
load each stage module. Feel free to modify the script to run in a distributed
way by distributing the for loop at [Note 3].
"""

import os
import torch
from torch.distributed.pipelining import pipeline, SplitPoint
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import AutoModelForCausalLM, AutoTokenizer

from load_weights import load_weights, init_buffers

# Grab the model in meta/fake mode
fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

with torch.device("meta"):
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf"
    )

llama.eval()
print(llama)

# Cast the model to FakeTensor with real device (from meta device) because
# there is autocast code in llama.  Autocast functions based on device of
# tensor. So we'd need to give it a real device instead of meta device.
with fake_mode:
    # [Note 1]: set device to "cuda" if you are using GPUs
    llama.to_empty(device="cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token
prompts = (
    "How do you", "I like to",
)

inputs = tokenizer(prompts, return_tensors="pt", padding=True)
real_ids = inputs["input_ids"]
# The example input needs to FakeTensor too
fake_ids = fake_mode.from_tensor(real_ids)

# Beginning of distributed
# [Note 2]: change world size here
world_size = 2
print(f"{world_size=}")

# Cut model by equal number of layers per rank
layers_per_rank = llama.config.num_hidden_layers // world_size
print(f"layers_per_rank = {layers_per_rank}")
split_spec = {
    f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
    for i in range(1, world_size)
}

# Convert model into a pipeline
pipe = pipeline(
    llama,
    mb_args=(fake_ids,),
    mb_kwargs={"output_attentions": False, "output_hidden_states": False, "use_cache": False,},
    split_spec=split_spec,
)

# Materialize each stage
# [Note 3]: remove this for loop if you are running this script in a
# distributed manner
for rank in range(world_size):
    stage_module = pipe.get_stage_module(rank)
    print(f"Loading weights into stage {rank}")
    load_weights(stage_module)
    if hasattr(llama, "buf_init_callbacks"):
        init_buffers(stage_module, "cpu", llama.buf_init_callbacks)
    stage_module.print_readable()

