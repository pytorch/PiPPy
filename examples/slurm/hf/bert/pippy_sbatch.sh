#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates

#SBATCH --job-name=bert_pippy

#SBATCH --open-mode=append

#SBATCH --partition=train

#SBATCH --nodes=2

#SBATCH --ntasks-per-node=8

#SBATCH --cpus-per-task=12

#SBATCH --gpus-per-node=8

#SBATCH --gpus-per-task=1

#SBATCH --time=1:00:00

srun --label pippy_wrapper.sh
