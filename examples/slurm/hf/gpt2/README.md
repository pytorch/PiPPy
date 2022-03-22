# PiPPy CUDA RPC Demo

Requires 2 nodes with 8 GPUs on slurm partion `train`.

Splits Huggingface GPT2 model into 14 submodules and runs it on 2 nodes with 8 GPUs each(16 workers total, 2 unused).

Run command:
```commandline
sbatch pippy_slurm.sh
```
Sample output:
```
 0: REPLICATE config: False -> MultiUseParameterConfig.TRANSMIT
 0: Instantiating GPT2 Pipeline
 0: /fsx/users/pbelevich/miniconda/envs/PiPPy37/lib/python3.7/site-packages/torch/fx/_symbolic_trace.py:420: UserWarning: Was not able to add assertion to guarantee correct inputs to specialized function. It is up to the user to make sure that your inputs match the inputs you specialized the function with.
 0:   "Was not able to add assertion to guarantee correct inputs to "
 0: /fsx/users/pbelevich/PiPPy/pippy/PipelineDriver.py:404: UserWarning: Running pipeline with 14 stages on world_size of 16. Remaining ranks will be idle.
 0:   warnings.warn(f'Running pipeline with {len(executor_descriptors)} stages on world_size of {self.world_size}. '
 0: Running GPT2 pipeline. NB: if this is too slow, set OMP_NUM_THREADS to a higher value
 0: Running reference pipeline
 0: /fsx/users/pbelevich/miniconda/envs/PiPPy37/lib/python3.7/site-packages/torch/testing/_deprecated.py:35: FutureWarning: torch.testing.assert_allclose() is deprecated since 1.12 and will be removed in 1.14. Use torch.testing.assert_close() instead. For detailed upgrade instructions see https://github.com/pytorch/pytorch/issues/61844.
 0:   warnings.warn(msg, FutureWarning)
 0: equivalence test passed 1.52587890625e-05 ref 1.52587890625e-05
 0: profiling run completed 0.0001678466796875 ref 1.52587890625e-05
```