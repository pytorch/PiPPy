# Copyright (c) Meta Platforms, Inc. and affiliates
set -ex
python test/min_gpt_tracing.py
pytest --cov=pippy test
REPLICATE=0 test/launch_local_test_forward.sh
REPLICATE=1 test/launch_local_test_forward.sh
REPLICATE=0 test/launch_local_test_forward_backward.sh
REPLICATE=1 test/launch_local_test_forward_backward.sh
