This example demonstrates how to create a pipeline based on a model on CPU, move different parts of the model to GPU
and run the model with data on GPU. This technique can help when a model is too large to materialize on a single GPU.

Run command:
```
$ torchrun --nproc-per-node 4 gpt2_cpu_init.py
```
