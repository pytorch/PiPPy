#!/bin/bash

nvidia-smi
nvcc --version
which python3
python3 --version
yum install -y python3-pip
which pip3
pip3 --version

# Print test options
echo "VERBOSE: ${VERBOSE}"
echo "REPLICATE: ${REPLICATE}"
echo "SCHEDULE: ${SCHEDULE}"

# Install dependencies
pip3 install flake8 pytest pytest-cov numpy
if [ -f requirements.txt ]; then pip3 install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html; fi

# Install pavel's huggingface fork
pip3 install git+https://github.com/pbelevich/transformers.git@compatible_with_pt_master sentencepiece

# Install pippy
python3 setup.py install

# Run all integration tests
python3 test/local_test_forward.py --replicate ${REPLICATE} -s ${SCHEDULE}
python3 test/local_test_forward_backward.py --replicate ${REPLICATE} -s ${SCHEDULE}
python3 examples/slurm/hf/gpt2/pippy_gpt2.py --replicate ${REPLICATE} -s ${SCHEDULE}

# Run flaky integration tests
python3 test/local_test_forward_hf_gpt2.py --replicate ${REPLICATE} -s ${SCHEDULE} || true
python3 test/local_test_forward_hf_bert.py --replicate ${REPLICATE} -s ${SCHEDULE} || true
