## TP Inference of Llama

Here we convert the Meta Llama model, which is based on Fairscale TP to PT-D compliant checkpoints and use PT TP (DTensor) API to run the Distributed inference.


### How to run the example



1- Make sure you have access to llama weights on [HF model hub](https://huggingface.co/meta-llama), there is form you need to fill up and within few mins you will get access. ANy model name on the hub **without -hf** is Meta/FAIR weight.

Make sure you you are signed up in HF as well, you will need you API token than can be access from [here](https://huggingface.co/settings/tokens), note to use the same email for accessing the weights as email you signed in to HF.

Once you have the access, in your terminal login to HF

```
huggingface-cli login YOUR_TOKEN

```

2- install requirements

```
pip install transformers fairscale
```

3- Download Meta llama weights from HF hub, it will download the models into `model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824/`

```
python download.py --model_name meta-llama/Llama-2-7b

```

4- convert the checkpoints to  PT-D compliant checkpoints as follows, note that for 7B `model_parallel_size 1` for 13B would be `model_parallel_size 2` and 70B `model_parallel_size 8`.

```
torchrun --nnodes 1 --nproc_per_node 8 convert_checkpoints.py --original_ckpt_dir  ./model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824/ --tokenizer_path ./model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824/tokenizer.model --model_parallel_size 1 --save_checkpoint_dir converted_checkpoints

```



5- Run the inference and generate tokens:

```
torchrun --nnodes 1 --nproc_per_node 8 generate.py  --ckpt_dir  ../model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824/ --tokenizer_path ../model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824/tokenizer.model --model_parallel_size 1 --converted_ckpt_dir converted_checkpoints

```



The following will run try to

a- first [build llama without any TP with meta device](https://github.com/pytorch/PiPPy/blob/2d_inference/examples/inference/tp_inference/llama2.py#L491) --> `meta_model` 

b- use the `meta_model` to [convert Meta/FAIR weights to PT-D compliant checkpoints](https://github.com/pytorch/PiPPy/blob/2d_inference/examples/inference/tp_inference/llama2.py#L433)

c- TP(lize) `meta_model` with [PT-D TP ](https://github.com/pytorch/PiPPy/blob/2d_inference/examples/inference/tp_inference/llama2.py#L499)

4- [load the PT-D compliant checkpoints to TP(lized)](https://github.com/pytorch/PiPPy/blob/2d_inference/examples/inference/tp_inference/llama2.py#L437) `meta_model` 

```
torchrun --nnodes 1 --nproc_per_node 8 convert_checkpoints.py --ckpt_dir  ../model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824/ --tokenizer_path ../model/models--meta-llama--Llama-2-7b/snapshots/365ffa8f1a6c455d3e2028ae658236b4b85ba824/tokenizer.model --model_parallel_size 1

```
