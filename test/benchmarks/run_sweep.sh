#!/bin/bash

python pippy_decoders_benchmarks.py  --model_name 'facebook/opt-6.7b' --batch-size 1 --chunks 1 --log_finename 'pippy_half.csv'


# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 2 --chunks 1
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 2 --chunks 2

# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 4 --chunks 1
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 4 --chunks 2
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 4 --chunks 4

# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 8 --chunks 1
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 8 --chunks 2
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 8 --chunks 4


# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 8 --chunks 1
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 8 --chunks 2
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 8 --chunks 4


# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 16 --chunks 1
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 16 --chunks 2
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 16 --chunks 4


# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 32 --chunks 1
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 32 --chunks 2
# python pippy_decoders_benchmarks.py --model_name 'facebook/opt-30b' --batch-size 32 --chunks 4


python pippy_decoders_benchmarks.py --batch-size 64 --chunks 1
python pippy_decoders_benchmarks.py --batch-size 64 --chunks 2
python pippy_decoders_benchmarks.py --batch-size 64 --chunks 4

python pippy_decoders_benchmarks.py --batch-size 128 --chunks 1
python pippy_decoders_benchmarks.py --batch-size 128 --chunks 2
python pippy_decoders_benchmarks.py --batch-size 128 --chunks 4