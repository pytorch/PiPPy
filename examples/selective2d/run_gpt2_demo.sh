#!/bin/bash
NUM_GPUS=$1

# GPT2 demo
torchrun --nproc-per-node 8 gpt2_demo.py --pp_size 4 --inner_cut --n_layer 8 --debug --inference 2>&1 | tee gpt2-inference.log
