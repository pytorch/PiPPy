#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export LOCAL_RANK=${SLURM_LOCALID}
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}

# Small
python -u pippy_gpt2.py

# 33B
#python -u pippy_gpt2.py --checkpoint=1 --gspmd=1 \
#  --n_layer=96 \
#  --n_head=96 \
#  --n_embd=5376
