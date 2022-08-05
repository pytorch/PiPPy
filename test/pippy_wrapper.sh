#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export LOCAL_RANK=${SLURM_LOCALID}
# Optional: depending on whether the application wants each procoess to see only 1 GPU or all GPUs
#export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}
export VERBOSE=1

python -u "$@" 2>&1
