#!/bin/bash

set -x

# Print test options
echo "VERBOSE: ${VERBOSE}"
echo "REPLICATE: ${REPLICATE}"
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

# Install pavel's huggingface fork
pip3 install git+https://github.com/huggingface/transformers.git@main sentencepiece

# Install pippy
python3 setup.py install

# Run all integration tests
python3 test/local_test_forward.py --replicate ${REPLICATE} -s ${SCHEDULE}
python3 test/local_test_forward_backward.py --replicate ${REPLICATE} -s ${SCHEDULE}
python3 examples/slurm/hf/gpt2/pippy_gpt2.py --replicate ${REPLICATE} -s ${SCHEDULE}

# Run flaky integration tests
python3 test/local_test_forward_hf_gpt2.py --replicate ${REPLICATE} -s ${SCHEDULE} || true
python3 test/local_test_forward_hf_bert.py --replicate ${REPLICATE} -s ${SCHEDULE} || true
