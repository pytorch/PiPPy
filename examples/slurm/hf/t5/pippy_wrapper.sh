#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export LOCAL_RANK=${SLURM_LOCALID}
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}

python -u pippy_t5.py \
  --model_config=t5_3b_config.json \
  --dp_group_size="${SLURM_JOB_NUM_NODES}" \
  --record_mem_dumps=0 \
  --checkpoint=1
