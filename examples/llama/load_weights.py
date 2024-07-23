# Copyright (c) Meta Platforms, Inc. and affiliates

import json
import torch

from typing import Optional


def load_weights(
    stage_module: torch.nn.Module,
    weight_index_file: Optional[str] = "pytorch_model.bin.index.json",
):
    """
    Load weights from Hugging Face checkpoints into a stage module.

    This is a utility for Hugging Face ModelHub checkpoints that comes with an
    index file and multiple binary files.  The index file indicates which
    parameter is saved in which binary. An example can be found at:
    https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main

    Please download the following files in the same directory as this script:
    - pytorch_model.bin.index.json
    - pytorch_model-00001-of-00002.bin
    - pytorch_model-00002-of-00002.bin
    """

    state_dict = stage_module.state_dict()
    updated_states = dict()

    # Get the weight map -- a map from parameter name to file it is saved in
    f = open(weight_index_file)
    js = json.load(f)
    weight_map = js["weight_map"]

    # Figure the set of binary files we'd need to open in order to fill the
    # state dict of the stage module. It will be a subset of all the binary
    # files because the stage module is a partition of the full model.
    needed_files = set()
    for param in state_dict.keys():
        file = weight_map[param]
        needed_files.add(file)

    # Now we load the needed binary files
    for file in needed_files:
        checkpoint = torch.load(file, weights_only=True)
        for param in state_dict.keys():
            if weight_map[param] == file:
                state_dict[param] = checkpoint[param]
                updated_states.setdefault(param, None)

    # Check if the module's state dict will be fully updated from checkpoint
    if state_dict.keys() == updated_states.keys():
        print("Fully updated state dict")
    else:
        print("Partially updated state dict")

    # Now load the weights into the stage module
    # We use `assign=True` because otherwise the properties of the tensors in
    # the current module are preserved.
    stage_module.load_state_dict(state_dict, assign=True)

