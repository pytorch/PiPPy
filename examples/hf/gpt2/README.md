# PiPPy CUDA RPC Demo

Requires 2 nodes with 8 GPUs on slurm partition `train`.

Splits Huggingface GPT2 model into 14 submodules and runs it on 2 nodes with 8 GPUs each(16 workers total, 2 unused).

Run command:
```commandline
PiPPy/examples$ sbatch pippy_sbatch.sh python ./hf/gpt2/pippy_gpt2.py
```
Sample output:
```
10: rank = 10 host/pid = train-st-p3dn24xlarge-3/85935
 8: rank = 8 host/pid = train-st-p3dn24xlarge-3/85939
14: rank = 14 host/pid = train-st-p3dn24xlarge-3/85937
13: rank = 13 host/pid = train-st-p3dn24xlarge-3/85932
11: rank = 11 host/pid = train-st-p3dn24xlarge-3/85940
 0: rank = 0 host/pid = train-st-p3dn24xlarge-2/66387
 9: rank = 9 host/pid = train-st-p3dn24xlarge-3/85934
 7: rank = 7 host/pid = train-st-p3dn24xlarge-2/66384
12: rank = 12 host/pid = train-st-p3dn24xlarge-3/85933
 2: rank = 2 host/pid = train-st-p3dn24xlarge-2/66386
 6: rank = 6 host/pid = train-st-p3dn24xlarge-2/66385
15: rank = 15 host/pid = train-st-p3dn24xlarge-3/85938
 4: rank = 4 host/pid = train-st-p3dn24xlarge-2/66391
 5: rank = 5 host/pid = train-st-p3dn24xlarge-2/66389
 1: rank = 1 host/pid = train-st-p3dn24xlarge-2/66388
 3: rank = 3 host/pid = train-st-p3dn24xlarge-2/66390
 0: REPLICATE config: False -> MultiUseParameterConfig.TRANSMIT
 0: Instantiating GPT2 Pipeline
 0: /fsx/users/pbelevich/miniconda/envs/PiPPy39/lib/python3.9/site-packages/transformers/modeling_utils.py:2327: UserWarning: Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod. (Triggered internally at  ../torch/csrc/utils/python_arg_parser.cpp:225.)
 0:   x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
 0: /fsx/users/pbelevich/miniconda/envs/PiPPy39/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py:193: UserWarning: Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod. (Triggered internally at  ../torch/csrc/utils/python_arg_parser.cpp:225.)
 0:   attn_weights = torch.matmul(query, key.transpose(-1, -2))
 0: /fsx/users/pbelevich/miniconda/envs/PiPPy39/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py:206: UserWarning: Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod. (Triggered internally at  ../torch/csrc/utils/python_arg_parser.cpp:225.)
 0:   attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
 0: /fsx/users/pbelevich/miniconda/envs/PiPPy39/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py:222: UserWarning: Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod. (Triggered internally at  ../torch/csrc/utils/python_arg_parser.cpp:225.)
 0:   attn_output = torch.matmul(attn_weights, value)
 0: /fsx/users/pbelevich/miniconda/envs/PiPPy39/lib/python3.9/site-packages/transformers/activations.py:34: UserWarning: Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod. (Triggered internally at  ../torch/csrc/utils/python_arg_parser.cpp:225.)
 0:   return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
 0: /fsx/users/pbelevich/PiPPy/pippy/PipelineDriver.py:471: UserWarning: Running pipeline with 14 stages on world_size of 16. Remaining ranks will be idle.
 0:   warnings.warn(f'Running pipeline with {len(executor_descriptors)} stages on world_size of {self.world_size}. '
 0: Running GPT2 pipeline. NB: if this is too slow, set OMP_NUM_THREADS to a higher value
 0: Running reference pipeline
 0: /fsx/users/pbelevich/miniconda/envs/PiPPy39/lib/python3.9/site-packages/torch/testing/_deprecated.py:35: FutureWarning: torch.testing.assert_allclose() is deprecated since 1.12 and will be removed in 1.14. Use torch.testing.assert_close() instead. For detailed upgrade instructions see https://github.com/pytorch/pytorch/issues/61844.
 0:   warnings.warn(msg, FutureWarning)
 0: equivalence test passed -0.00017547607421875 ref -0.00017547607421875
 0: profiling run completed -0.00072479248046875 ref -0.00017547607421875
```
