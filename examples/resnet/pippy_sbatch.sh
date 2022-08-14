#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates

#SBATCH --job-name=mnist_pippy

#SBATCH --open-mode=append

#SBATCH --partition=train

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=8

#SBATCH --cpus-per-task=12

#SBATCH --gpus-per-node=8

#SBATCH --time=1:00:00

srun --label pippy_wrapper.sh
