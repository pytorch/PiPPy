#!/bin/bash
NUM_GPUS=$1

torchrun --nproc-per-node $NUM_GPUS mlp_demo.py --pp_size 4 --inner_cut --n_layer 8 --debug --inference 2>&1 | tee mlp-inference.log
