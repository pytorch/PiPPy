#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates

#SBATCH --job-name=t5_pippy

#SBATCH --open-mode=append

#SBATCH --partition=train

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=8

#SBATCH --cpus-per-task=12

#SBATCH --gpus-per-node=8

#SBATCH --time=1:00:00

# Use the following settings instead if using double pipes
# as it uses 2x number of processes
##SBATCH --ntasks-per-node=16
##SBATCH --cpus-per-task=6
##SBATCH -m plane=8

srun --label pippy_wrapper.sh
