# Copyright (c) Meta Platforms, Inc. and affiliates
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
torchrun --standalone --nnodes=1 --nproc_per_node=10 "${SCRIPTPATH}/local_test_forward.py"
