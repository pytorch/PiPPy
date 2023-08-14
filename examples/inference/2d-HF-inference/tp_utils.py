import re
import torch
import torch.nn as nn

from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )
from pippy.IR import annotate_split_points, PipeSplitWrapper

def find_mlp_layers(model):
    linear_layers = []
    # Check if the model has an decoder attribute (like OPT, GPT, llama etc.)
    num_layer=0
    for name, module in model.named_modules():
        num_layer+=1
        if isinstance(module, nn.Linear):
            linear_layers.append(name)

    # Add more conditions for other architectures as needed
    mlp_layers = [s for s in linear_layers if not "attn" in s]
    print("=========== num_layer", num_layer)
    return mlp_layers

def find_mlp_layers_pattern(layer_names):
    # Define a pattern to match numbers
    pattern = re.compile(r"\d+")

    # Replace the number with a placeholder and store the result in a new list
    mlp_layer_names = set([pattern.sub("{i}", layer_name) for layer_name in layer_names])

    # Print the formatted strings
    for formatted_string in mlp_layer_names:
        print(formatted_string)
    return mlp_layer_names

def stage_mlp_layer_name(mlp_layer_names):
    stage_mlp_layer_names = [s.replace('.', '_') for s in mlp_layer_names]    
    
       
def parallelize_stage_MLP_block(model, twod_mesh):
        # block = model.get_submodule(module_path)
        parallelized_block = parallelize_module(
            module=model,
            device_mesh=twod_mesh,
            parallelize_plan={
                "model_decoder_layers_{i}_fc1": ColwiseParallel(),
                "model_decoder_layers_{i}_fc2": RowwiseParallel(),
            },
            tp_mesh_dim=1,
        )
        return parallelized_block
    


def parallelize_MLP_block(model, module_path, twod_mesh):
    block = model.get_submodule(module_path)
    parallelized_block = parallelize_module(
        module=block,
        device_mesh=twod_mesh,
        parallelize_plan={
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
            # "T5LayerFF": PairwiseParallel(),
        },
        tp_mesh_dim=1,
    )
    return parallelized_block

def parallelize_Attn_block(model, module_path, twod_mesh):
    block = model.get_submodule(module_path)
    parallelized_block = parallelize_module(
        module=block,
        device_mesh=twod_mesh,
        parallelize_plan={
            "k_proj": ColwiseParallel(),
            "v_proj": ColwiseParallel(),
            "q_proj": ColwiseParallel(),
            "out_proj": RowwiseParallel(),
            
            # "SelfAttention": PairwiseParallel(),
        },
        tp_mesh_dim=1,
    )
    return parallelized_block

def tp(model, mesh):
    for i in range(model.config.num_hidden_layers):
        block = parallelize_MLP_block(model, f"model.decoder.layers.{i}", mesh)
        # block = parallelize_Attn_block(model, f"model.decoder.layers.{1}.self_attn.k_proj", mesh)
        # block = parallelize_Attn_block(model, f"model.decoder.layers.{1}.self_attn.v_proj", mesh)
        # block = parallelize_Attn_block(model, f"model.decoder.layers.{1}.self_attn.q_proj", mesh)
        # block = parallelize_Attn_block(model, f"model.decoder.layers.{1}.self_attn.out_proj", mesh)

def parallelize_stage_llama_MLP_block(stage, num_layers,pp_group_size,pp_rank, twod_mesh):
    num_layer_per_rank = num_layers/pp_group_size
    start_range = int(num_layer_per_rank*pp_rank)
    end_range = int(num_layer_per_rank*(pp_rank+1))
    parallelize_plan={}
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

  
def tp_stage(stage,num_layers, mesh):

    for i in range(num_layers):
        try:
            block = parallelize_stage_llama_MLP_block('model_layers_{i}', mesh)
        except:
            pass


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