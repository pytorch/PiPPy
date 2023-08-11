#!/bin/bash
NUM_GPUS=$1

#no opt
#torchrun --nproc-per-node $NUM_GPUS mlp_demo.py --pp_size 4 --n_layer 8 --debug --inference --i_stage 1 --nstreams 1 --n_chunks 4 2>&1 | tee mlp-inference.log
#torchrun --nproc-per-node $NUM_GPUS mlp_demo.py --pp_size 4 --n_layer 8 --train_iters 50 --inference --i_stage 1 --nstreams 1 --n_chunks 4 2>&1 | tee mlp-inference.log

#ar1
#torchrun --nproc-per-node $NUM_GPUS mlp_demo.py --pp_size 4 --n_layer 8 --debug --inference --i_stage 1 --nstreams 2 --n_chunks 4 2>&1 | tee mlp-inference.log
#torchrun --nproc-per-node $NUM_GPUS mlp_demo.py --pp_size 4 --n_layer 8 --train_iters 50 --inference --i_stage 1 --nstreams 2 --n_chunks 4 2>&1 | tee mlp-inference.log

#ar1+ar2
#torchrun --nproc-per-node $NUM_GPUS mlp_demo.py --pp_size 4 --inner_cut --n_layer 8 --debug --i_stage 2 --n_chunks 4 --nstreams 2 --inference 2>&1 | tee mlp-inference.log
torchrun --nproc-per-node $NUM_GPUS mlp_demo.py --pp_size 4 --inner_cut --n_layer 8 --train_iters 50 --inference --i_stage 2 --nstreams 2 --n_chunks 4 --test 2>&1 | tee mlp-inference.log
