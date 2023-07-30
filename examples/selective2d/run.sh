#!/bin/bash
NUM_GPUS=8

# 1. run dimension solver
python3 ./dim_solver/main.py --num-gpu $NUM_GPUS --opt-ar1

# 2. process output file
# output file name: solver.out
# output data: pp_group_size tp_group_size size_microbatch i_stage n_chunks

pp_group_size=$(cat ./solver.out | awk '{print $1}')
tp_group_size=$(cat ./solver.out | awk '{print $2}')
size_microbatch=$(cat ./solver.out | awk '{print $3}')
i_stage=$(cat ./solver.out | awk '{print $4}')
n_chunks=$(cat ./solver.out | awk '{print $5}')

batch_size=$((size_microbatch * n_chunks))

# 3. run training with optimal configuration
torchrun --nproc-per-node=$NUM_GPUS 2d_train.py --batch-size=$size_microbatch --n_chunks $n_chunks --i_stage $i_stage
