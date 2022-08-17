#!/bin/bash

set -x

# Print test options
echo "VERBOSE: ${VERBOSE}"
echo "SHARD: ${SHARD}"

nvidia-smi
nvcc --version
cat /etc/os-release
which python3
python3 --version
which pip3
pip3 --version

# Install git
apt-get update
apt-get install git -y

# Install dependencies
# Turn off progress bar to save logs
pip3 install --upgrade pip
pip3 config set global.progress_bar off
pip3 install flake8 pytest pytest-cov pytest-shard numpy expecttest hypothesis
if [ -f requirements.txt ]; then pip3 install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html; fi

# Install pippy
python3 spmd/setup.py install

set -ex

# Run all integration tests
pytest --shard-id=${SHARD} --num-shards=4 --cov=spmd test/spmd/
