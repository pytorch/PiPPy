#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates

#SBATCH --job-name=t5_pippy

#SBATCH --open-mode=append

#SBATCH --partition=train

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=16

#SBATCH --cpus-per-task=6

#SBATCH --gpus-per-node=16

#SBATCH --time=1:00:00

srun --label pippy_wrapper.sh
