# $ torchrun --nproc-per-node 8 2d_llama.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pippy import Pipe, PipeSplitWrapper, annotate_split_points, PipelineStage
from torch.distributed._tensor import init_device_mesh


# Utility
def modify_view(
    gm: torch.fx.GraphModule,
    tp: int
):
    """
    Adjust dimension size of view ops to make them compatible with tensor
    parallelism.  For example, when TP is 4, we need to adjust `num_heads` from
    32 to 8.  This is needed for attention layers.
    """
    for node in gm.graph.nodes:
        if node.op == "call_method" and (
            node.target == "view" or node.target == "reshape"
        ):
            assert len(node.args) >= 4
            node.update_arg(3, node.args[3] // tp)
    gm.recompile()


# Grab the model
llama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

prompts = (
    "How do you", "I like to", "Can I help", "You need to",
    "The weather is", "I found a", "What is your", "You are so",
)  # bs = 8
tokenizer.pad_token = tokenizer.eos_token

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

pp_group_size = 2
tp_group_size = 4
mesh_2d = init_device_mesh("cuda", (pp_group_size, tp_group_size), mesh_dim_names=("pp", "tp"))
pp_group = mesh_2d["pp"].get_group()

llama.to(device).eval()
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

# Cut model by equal number of layers per rank
layers_per_stage = llama.config.num_hidden_layers // pp_group_size
for i in range(1, pp_group_size):
    annotate_split_points(llama,
        {f"model.layers.{i * layers_per_stage}": PipeSplitWrapper.SplitPoint.BEGINNING})

# Create a pipeline representation from the model
llama_pipe = Pipe.from_tracing(llama, pp_group_size, example_args=(inputs["input_ids"],))

# Create pipeline stage for each rank
stage_idx = rank // tp_group_size
stage = PipelineStage(llama_pipe, stage_idx, device=device, group=pp_group)

modify_view(stage.submod, tp_group_size)

# Tensor parallel
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
starting_layer = stage_idx * layers_per_stage
attn_plan = {}
mlp_plan = {}
for i in range(layers_per_stage):
    # HACK: the right fix is to remove the ".mod" added by PipeSplitWrapper
    extra = "_mod" if starting_layer > 0 and i == 0 else ""
    layer_name = f"L__self___model_layers_{starting_layer + i}{extra}"
    attn_plan.update({
        # Parallel self attention not working yet due to the dimension mismatch
        # after TP in view operation
        f"{layer_name}_self_attn_q_proj": ColwiseParallel(),
        f"{layer_name}_self_attn_k_proj": ColwiseParallel(),
        f"{layer_name}_self_attn_v_proj": ColwiseParallel(),
        f"{layer_name}_self_attn_o_proj": RowwiseParallel(),
    })
    mlp_plan.update({
        f"{layer_name}_mlp_gate_proj": ColwiseParallel(),
        f"{layer_name}_mlp_up_proj": ColwiseParallel(),
        f"{layer_name}_mlp_down_proj": RowwiseParallel(),
    })
tp_mesh = mesh_2d["tp"]
parallelize_module(
    stage.submod, tp_mesh, {**attn_plan, **mlp_plan}
)

# Run
if stage_idx == 0:
    args = inputs["input_ids"]
else:
    args = None
output = stage(args)

# Decode
if output is not None:
    next_token_logits = output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(next_token))
