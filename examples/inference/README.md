# PiPPy (Pipline Parallelism for PyTorch) Distributed Inference for Large Models

PiPPy helps to run very large models for inference by splitting the model into mutliple stages running on multiple GPUs.
PiPPy make this easier by providing a auto split API that automates this process for user.

## How It Works

PiPPy splits your model into multiple stages, each stage loaded on one gpu then the input batch will be furhter divided into micro-batches and run through the splits from
rank0..rankN. Results are being returned to rank0 as its runing the PipelineDriver. Please read more on pipleines [here](https://github.com/pytorch/tau/blob/main/README.md)

The flowchart below helps to visualize the process in high level as well.

<img src="https://user-images.githubusercontent.com/9162336/207237303-86dc02fe-dae0-4335-8d23-c56d31ecdb87.png" alt="drawing" width="400"/>

## PiPPy Supports Arbitary Model Partitioning

Unlike most of the available solutions that they need to know the model architecture beforehand, PiPPy supports arbitary PyTorch models.
* PiPPy supports both manual splitting and auto split.
* Auto split uses `split_policy` and support both `equal_size` and `threshod` policies, the name are self-explanatory.
* PiPPy use FX to trace and split the model.

## Settings To Care About

* **world_size** specifies your availble number of gpus for paritioning your model

* **split_policy** it can be either `equal_size`, `split_into_equal_size(number_of_workers)` or `threshod`, `split_on_size_threshold(#some number)`

* **schedule** for the pipline, we use `PipelineDriverFillDrain` for inference, please learn more about it [here](https://github.com/pytorch/tau/blob/main/README.md#advanced-pipeline-schedules).

* **chunks** it detemines the size of microbatches, microbatch = batch size/ chuncks

* **FX Tracers** use PiPPyHFTracer() is dealing with a HuggingFace model otherwise set to `None`

## Get the Pipeline Driver

After setting the above mentioned, to get your pipeline driver and access different stages of the model simply call below

```python
pipe_driver, stage_mode = pippy.all_compile(
            model,
            num_ranks=world_size,
            num_chunks=chunks,
            schedule="FillDrain",
            split_policy=split_policy,
            tracer=PiPPyHFTracer(),
            concrete_args=concrete_args,
        )
```
**Note** As PiPPY leverage FX tracing for partitioning, as a result for HuggingFace models that have `generate` method will need to call `inject_pipeline_forward(model, pipe_driver)` to  make `model.generate` available. This works for the decoder only  models so far, encoder-decoder models such as `T5` is in progress.

**Main difference between Pippy for training and inference is we dont need to call the init_data_parallel API in the inference. The reason is DDP init is only needed if we need backward pass which is not the case for inference.**



## HuggingFace Example

**Define a function such as run_all() and add the followings to it.**

We use a HuggingFace `OPT` model as the running example here. The `hf_generate.py` also support other models for text generation such as `Bloom`, `gpt2` and `codegen` family of the models as well. Make sure to specifiy the model name as follows ` python hf_generate.py --model_name "facebook/opt-2.7b" `. This is not limited to LLMs it also works for models such [RegNet 10B](https://huggingface.co/facebook/regnet-y-10b-seer).


* Load your model normally on CPU

example:

` model = AutoModelForCausalLM.from_pretrained('facebook/opt-6.7b', use_cache=False) `


*  Setup the model split policy

```python
from pippy import split_on_size_threshold, split_into_equal_size

if args.auto_split == "threshold":
        split_policy = split_on_size_threshold(490 * 1e6)
elif args.auto_split == "equal_size":
        split_policy = split_into_equal_size(number_of_workers)
```
* Make the concerete args (optional), If the model has inside an if-else condition, the concrete args can help FX determine which path to trace. For now control flow is not supported in FX tracing, we are working on integrating Torch Dynamo to make this more flexible. 

```python
inputs = tokenizer(prompt, return_tensors="pt")
input_names = inputs.keys()
sig = inspect.signature(model.forward)
concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
```

* Get the pipline driver and model stages with `pippy.all_compile()`. See the section above.

This under the hood, splits the model into a pipline, `Pipe.from_tracing` uses `torch.fx` symbolic tracing to turn our model into a directed acyclic graph (DAG) representation. Then, it groups together the operations and parameters into _pipeline stages_. Stages are represented as `submod_N` submodules, where `N` is a natural number. Note: here we use HF FX_tracer for tracing.

Loads to device directly using `defer_stage_init`, which basically let each rank trace the model and split the model and only materialize its own shard.

Finally, we get a `PipelineDriver` that runs the pipeline. It implements the runtime scheduling and communcation between stages.


* Run the inference by passing input data to the `PipelineDriver`.

`pipe_driver(**inputs)`


**we Now pass the run_all() function to the run_pippy() along with args to run the program**

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 8)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    args.gspmd = 1
    run_pippy(run_all, args)
```

To run the full example, simply run your Python inference script:

` python hf_generate.py --model_name 'facebook/opt-6.7b'' `

or

` torchrun --nproc_per_node=8 hf_generate.py --model_name 'facebook/opt-6.7b' `

### Run OPT model example

This has been tested for [OPT 2.7 and 30B](https://huggingface.co/facebook/opt-30b) on 8 V100 GPUs.

` python hf_generate.py --model_name 'facebook/opt-30b' `

### Run Bloom model example

This has been tested for [Bloom 3b](https://huggingface.co/docs/transformers/model_doc/bloom) on 8 V100 GPUs.

` python hf_generate.py --model_name 'bigscience/bloom-3b' `

### More models to try
-  "gpt2"
- "bigscience/bloom-3b"
- EleutherAI/gpt-neo-2.7B
- Salesforce/codegen-2B-multi

### Handling big models

If the model in HuggingFace has `pytorch_model.bin.index.json`, then you can use distributed loading to reduce the CPU memory.

For example, if you want to run [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1) with low CPU memory usage:

```
git lfs install
git clone https://huggingface.co/bigscience/bloom-7b1

torchrun --nproc_per_node 4 hf_generate.py --world_size 4 --model_name ./bloom-7b1 --index_filename bloom-7b1/pytorch_model.bin.index.json
```

In this case, each rank will only load a part of the model.