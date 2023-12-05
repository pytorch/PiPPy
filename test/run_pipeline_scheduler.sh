# To run samples:
# bash run_example.sh {num_gpus}{file_to_run.py} 
# file_to_run = example to launch.  Default = 'test_pipeline_schedule.py'
# num_gpus = num local gpus to use (must be at least 2). Default = 8
# schedule = --schedules
#echo "Launching ${2:-test_pipeline_schedule.py} with ${1:-8} gpus"
torchrun --nnodes=1 --nproc_per_node 8 --rdzv_endpoint="localhost:59123" test_pipeline_schedule.py --profiler True
#torchrun --nnodes=1 --nproc_per_node=${1:-8} --rdzv_id=101 --rdzv_endpoint="localhost:5972" ${2:-test_pipeline_schedule.py}