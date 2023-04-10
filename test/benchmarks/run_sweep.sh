#!/bin/bash

python pippy_decoders_benchmarks.py -batch-size 1 --chunks 1


python pippy_decoders_benchmarks.py -batch-size 2 --chunks 1
python pippy_decoders_benchmarks.py -batch-size 2 --chunks 2


python pippy_decoders_benchmarks.py -batch-size 8 --chunks 1
python pippy_decoders_benchmarks.py -batch-size 8 --chunks 2
python pippy_decoders_benchmarks.py -batch-size 8 --chunks 4


python pippy_decoders_benchmarks.py -batch-size 8 --chunks 1
python pippy_decoders_benchmarks.py -batch-size 8 --chunks 2
python pippy_decoders_benchmarks.py -batch-size 8 --chunks 4


python pippy_decoders_benchmarks.py -batch-size 16 --chunks 1
python pippy_decoders_benchmarks.py -batch-size 16 --chunks 2
python pippy_decoders_benchmarks.py -batch-size 16 --chunks 4


python pippy_decoders_benchmarks.py -batch-size 32 --chunks 1
python pippy_decoders_benchmarks.py -batch-size 32 --chunks 2
python pippy_decoders_benchmarks.py -batch-size 32 --chunks 4


python pippy_decoders_benchmarks.py -batch-size 64 --chunks 1
python pippy_decoders_benchmarks.py -batch-size 64 --chunks 2
python pippy_decoders_benchmarks.py -batch-size 64 --chunks 4

python pippy_decoders_benchmarks.py -batch-size 128 --chunks 1
python pippy_decoders_benchmarks.py -batch-size 128 --chunks 2
python pippy_decoders_benchmarks.py -batch-size 128 --chunks 4