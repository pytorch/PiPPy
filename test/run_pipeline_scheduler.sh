# To run samples:
# launcher for testing pipeline schedules
torchrun --nnodes=1 --nproc_per_node 8 --rdzv_endpoint="localhost:59124" test_pipeline_schedule.py