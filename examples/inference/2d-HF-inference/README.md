# HuggingFace 2D(TP+ PP) inference 

The idea here is to combine TensorParallel (TP) with Pipeline Parallelism (PP). At this point we still rely on FX tracing to partition the model. 

## Quick try

```bash
torchrun --nnodes 1 --nproc_per_node 8 pp_tp_inferece.py --model_name meta-llama/Llama-2-7b-chat-hf  --world_size 8 --pp_group_size 1 --tp_group_size 8

```
## How does 2D work here?

1- `compile_stage()` API takes a model/HuggingFace model (HF) it will partition the model to stages, where each stage mean the partition (e.g. few model layers) of the model on each rank, that has a send and receive method for handling data in/out of the stage.

```python 
 stages = pippy.compile_stage(
        model,
        rank=pp_rank,
        num_ranks= num_ranks,
        num_chunks=args.chunks,
        device=args.device,
        example_inputs=[input_ids],
        group=pp_group,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
    )
```

2- TP(lize) each stage, here we need to figure out the layers to TP(lize) them in the model ( I am working on automating this process for HF models). We first focus on MLP layers and will extend to Attention layer in the next step.

```python

def parallelize_stage_llama_MLP_block(stage, num_layers,pp_group_size,pp_rank, twod_mesh):
    num_layer_per_rank = num_layers/pp_group_size
    start_range = int(num_layer_per_rank*pp_rank)
    end_range = int(num_layer_per_rank*(pp_rank+1))
    for i in range(start_range,end_range):
        parallelize_plan={
                        f"model_layers_{i}_mlp_up_proj": ColwiseParallel(),
                        f"model_layers_{i}_mlp_gate_proj":ColwiseParallel(),
                        f"model_layers_{i}_mlp_down_proj": RowwiseParallel(),
                        
                        
                    }
    try :
        parallelized_block = parallelize_module(
            module=stage.submod,
            device_mesh=twod_mesh,
            parallelize_plan=parallelize_plan,
            tp_mesh_dim=1,
        )
    except:
        pass
```

3- Run the inference

```python

   if pp_rank == 0:
        out = stage(input_ids)
    elif pp_rank == args.pp_group_size - 1:
        out = stage()
    else:
        out= stage()
```

## What are the gaps at the moment/Next steps:

1- For now, we are moving the model to device before `compile_stage`, as compile stage today does not move stages to device. This need to be fixed by loading the model with meta device and then loading the checkpoints.

2- Need to figure out how, stages that represent the whole model work with generate() ( generate is wrapper that run the model.forward in a loop and each time append the generated token to the previously generated tokens and run forward again util reach to max_new_token limit).

Possible solutions :
    a- hook the stages to model.forward
    b- write a general generate function for PiPPy to handle it with stages.

3- Auto_Split policy does not work with compile_stage, for now replaced with even_cut as shown below.

```python

def even_cut(model, num_layer, pp_size):
        """
        Evenly cut a model into pp_size stages
        """
        cut = {}
        cutpoint = num_layer // pp_size
        for i in range(num_layer):
            name = f"model.layers.{i}"
            if i > 0 and i % cutpoint == 0:
                cut[name] = PipeSplitWrapper.SplitPoint.END  # or END
        annotate_split_points(model, cut) 

```