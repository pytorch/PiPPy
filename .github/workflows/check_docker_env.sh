#!/bin/bash

nvidia-smi
nvcc --version
which python
python --version
which pip
pip --version

# Print test options
echo "VERBOSE: ${VERBOSE}"
echo "REPLICATE: ${REPLICATE}"
echo "SCHEDULE: ${SCHEDULE}"

# Install dependencies
pip install flake8 pytest pytest-cov numpy
if [ -f requirements.txt ]; then pip install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html; fi

# Install pavel's huggingface fork
pip install git+https://github.com/pbelevich/transformers.git@compatible_with_pt_master sentencepiece

# Install pippy
python setup.py install

# Run all integration tests
python test/local_test_forward.py --replicate ${REPLICATE} -s ${SCHEDULE}
python test/local_test_forward_backward.py --replicate ${REPLICATE} -s ${SCHEDULE}
python examples/slurm/hf/gpt2/pippy_gpt2.py --replicate ${REPLICATE} -s ${SCHEDULE}

# Run flaky integration tests
python test/local_test_forward_hf_gpt2.py --replicate ${REPLICATE} -s ${SCHEDULE} || true
python test/local_test_forward_hf_bert.py --replicate ${REPLICATE} -s ${SCHEDULE} || true
