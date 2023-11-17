#!/bin/bash

set -x

# Print test options
echo "SCHEDULE: ${SCHEDULE}"

nvidia-smi
nvcc --version
which python3
python3 --version
which pip3
pip3 --version

# Install git
apt-get update
apt-get install git -y

# Install dependencies
# Turn off progress bar to save logs
pip3 config set global.progress_bar off
pip3 install flake8 pytest pytest-cov numpy
if [ -f requirements.txt ]; then pip3 install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html; fi

# Install pippy
python3 setup.py install

set -ex

# Run all integration tests
torchrun --nproc-per-node 4 test_fwd.py -s ${SCHEDULE}
torchrun --nproc-per-node 4 test_bwd.py -s ${SCHEDULE}
