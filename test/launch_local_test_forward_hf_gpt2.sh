SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
torchrun --standalone --nnodes=1 --nproc_per_node=15 "${SCRIPTPATH}/local_test_forward_hf_gpt2.py"
