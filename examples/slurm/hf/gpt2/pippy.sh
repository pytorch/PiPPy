#!/bin/bash

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export LOCAL_RANK=${SLURM_LOCALID}
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}

python -u pippy_cuda_rpc.py
