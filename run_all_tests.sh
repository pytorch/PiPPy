#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates

set -ex
python test/min_gpt_tracing.py
pytest --cov=pippy test

for CUDA in 0 1; do
  for REP in 0 1; do
    for SCHD in FillDrain 1F1B; do
      python test/local_test_forward.py -s ${SCHD} --replicate ${REP} --cuda ${CUDA}
      python test/local_test_forward_backward.py -s ${SCHD} --replicate ${REP} --cuda ${CUDA}
    done
  done
done
