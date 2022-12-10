# PiPPY distributed inference for large models

PiPPY helps to run very large models for inference by splitting the model into mutliple stages running on multiple GPUs.
PiPPY make this easier by providing a auto split API that automates this process for user. 

## How it works

PiPPY splits your model into multiple stages, each stage loaded on one gpu then the input batch will be furhter divided into micro-batches and run through the splits from 
rank0..rankN. Results are being returned to rank0 as its runing the PipelineDriver. Please read more on pipleines [here](https://github.com/pytorch/tau/blob/main/README.md)

The flowchart below helps to visualize the process in high level as well.

<img src="https://user-images.githubusercontent.com/9162336/206815839-60d43f93-ff4b-4a5e-99db-1e0c4a55a4ac.png" alt="drawing" width="400"/>


## PiPPY support arbitary checkpoint splitting 

Unlike most of the available solutions that they need to know the model architecture beforehand, PiPPY supports arbitary PyTorch checkpoints.
* PiPPY supports both manual splitting and auto split.
* Auto split uses `split_policy` and support both `equal_size` and `threshod` policies, the name are self-explanatory.
* PiPPY use FX to trace and split the model.

## Setting you need to care about

* pp_group_size configure the number of pipeline parallelism group, meaning essentially on how many gpus our model to need be splitted and form a pipeline.

**Main difference between Pippy for training and inference is we dont need to call the init_data_parallel API in the inference. The reason is DDP init is only needed if we need backward pass which is not the case for inference.**


For example to serve two copies of the model with 8 gpus, assuming we can serve a full copy of a model splitted on 4 gpus, we set the pp_group_size=4. In this case the pipeline driver is running from rank 0 for the first copy and rank4 for the second copy of the model. In the run_master funtion after initializing the stage after splitting `model_pipe.defer_stage_init(args.device)' need to check if the rank=!0 or 4 to return. As shown below:
```
model_pipe.defer_stage_init(args.device)
torch.distributed.barrier(args.pp_group)
if args.rank!=0:
        return 
```
        
Then the resulted models (2 copies) will look like:

<img src="https://user-images.githubusercontent.com/9162336/206811740-1d5d4d7a-7018-4e9b-8ab7-cad030e2981a.png" alt="drawing" width="400"/>




## How to Use PiPPY for inference

**Define a function such as run_master() and add the followings to it.**

We use a HuggingFace T5 model as the running example here. The `HF_inference.py` also support HF OPT, Bloom, RegNet models as well. Make sure to specifiy the model name as follows ` python HF_inference.py --model_name "facebook/opt-2.7b" `

* Load your model normally on CPU

example:

` t5 = AutoModelForSeq2SeqLM.from_pretrained('t5-11b', use_cache=False) `

* The "MULTI_USE_PARAM_CONFIG" addresses corner cases where if an Operation would need a parameter that has been placed on a different GPU. In this case with setting we can either REPLICATE or TRANSMIT that paramter to the GPU/rank that needs it. 

 `MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if args.replicate else MultiUseParameterConfig.TRANSMIT`

*  Setup the model split policy

```
from pippy import split_on_size_threshold, split_into_equal_size

if args.auto_split == "threshold":
        split_policy = split_on_size_threshold(490 * 1e6)
elif args.auto_split == "equal_size":
        split_policy = split_into_equal_size(number_of_workers)
```
* Make the concerete args, this is required for FX tracing by giving it concerete inputs, this will be used to run the input through the model to check if there any control flow. For now control flow is not supported in FX tracing, we are working on integrating Torch Dynamo to make this more flexible. 

```
t5_input_dict = {'input_ids': inp, 'decoder_input_ids': inp}
input_names = t5_input_dict.keys()
sig = inspect.signature(t5.forward)
concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
```

* Split the model into a pipline, `Pipe.from_tracing` uses `torch.fx` symbolic tracing to turn our model into a directed acyclic graph (DAG) representation. Then, it groups together the operations and parameters into _pipeline stages_. Stages are represented as `submod_N` submodules, where `N` is a natural number. Note:here we use HF FX_tracer for tracing.

```
from PiPPY.IR import Pipe

t5_pipe = Pipe.from_tracing(t5, MULTI_USE_PARAM_CONFIG, tracer=PiPPYHFTracer(), concrete_args=concrete_args,
                                output_loss_value_spec=None, split_policy=split_policy
                                )
```

* Set the number of chunks that decide the microbatch sizes

```
all_worker_ranks = pp_ranks[PiPPY.utils.exclude_master:PiPPY.utils.exclude_master + number_of_workers]
chunks = args.chunks or len(all_worker_ranks)
batch_size = args.batch_size * chunks

```
* Load to device directly using "defer_stage_init", which basically let each rank trace the model and split the model and only materialize its own shard
The barrier would make sure all the rank have loaded their shards and finally we make sure that only rank0 run the pipe.

```
 t5_pipe.defer_stage_init(args.device)
 PiPPY.utils.pp_group_barrier()
 if args.rank!=0:
        return 
 ```

* Define the input/ouput to the model, "TensorChunkSpec(0)" below shows the batch dimension in the input is zero. Pipelining relies on _micro-batching_--that is--the process of dividing the program's input data into smaller chunks and feeding those chunks through the pipeline sequentially. Doing this requires that the data and operations be _separable_, i.e.there should be at least one dimension along which data can be split such that the program does not have interactions across this dimension.

```
kwargs_chunk_spec = {'input_ids': TensorChunkSpec(0), 'decoder_input_ids': TensorChunkSpec(0)}

output_chunk_spec = {"logits": TensorChunkSpec(0),"encoder_last_hidden_state": TensorChunkSpec(0)}

```
* Choose an schedule for the pipline, we use "PipelineDriverFillDrain" here, please learn more about it [here](https://github.com/pytorch/tau/blob/main/README.md#advanced-pipeline-schedules). For inference we would need only "PipelineDriverFillDrain".

```
schedules = {
    'FillDrain': PipelineDriverFillDrain,
}
```
* Now we have all the settings lets define the PipelineDriver that runs the pipeline. To learn more about different schedules for piplelines please use this link[]

```
pipe_driver: PipelineDriverBase = schedules[args.schedule](t5_pipe, chunks, args_chunk_spec, kwargs_chunk_spec,
                                                            output_chunk_spec,
                                                            world_size=len(all_worker_ranks),
                                                            all_ranks=all_worker_ranks,
                                                            checkpoint=bool(args.checkpoint), #optional in case pretrained weights not required
                                                            )
```

* Run the inference by passing input data to the PipelineDriver.

`pipe_driver(**t5_input_dict)`


**we need to pass the run_master() function to the run_PiPPY() along with args to run the pipeline**

* Here we need to make sure args.gspmd is set that will let run_PiPPY() to let each rank do the trace and sharding.

```

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 8)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    args.gspmd = 1
    run_pippy(run_master, args)

```
Then simply run your python inference script

` python HF_inference.py --model_name 't5-11b' `

### Run OPT model example

This has been tested for [OPT 2.7 and 30B](https://huggingface.co/facebook/opt-30b) on 8 V100 GPUs.

` python HF_inference.py --model_name 'facebook/opt-30b' `

### Run Bloom model example

This has been tested for [Bloom 3b](https://huggingface.co/docs/transformers/model_doc/bloom) on 8 V100 GPUs.

` python HF_inference.py --model_name 'bigscience/bloom-3b' `

### Run RegNet Vision model example

This has been tested for [RegNet 10B](https://huggingface.co/facebook/regnet-y-10b-seer) on 8 V100 GPUs.

` python HF_inference.py --model_name 'facebook/regnet-y-10b-seer' `
