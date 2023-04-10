# Benchmark PiPPy for Inference

### Benchmark PiPPY for HuggingFace Decoder models

To benchmark the HuggingFace(HF) models with PiPPY for Inference run the following

```bash
python pippy_decoders_benchmarks.py --model_name -batch-size 1 --chunks 1 --log-filename 'CSV_filename'

```
Results are logged into a `csv` file passed through `--log-filename `.

To sweep over different batch size and chunks you can run, make sure the args to what you prefer to benchmark.

```bash
sh run_sweep.sh

```
