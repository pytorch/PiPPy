# Copyright (c) Meta Platforms, Inc. and affiliates
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
torchrun --standalone --nnodes=1 --nproc_per_node=15 "${SCRIPTPATH}/local_test_forward_hf_bert.py" $@
