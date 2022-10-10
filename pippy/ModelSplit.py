# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Dict, List, Tuple

import torch
import pippy.fx
from pippy.IR import pipe_split

"""
Analyze size of parameters/buffers used by each node in the graph
Here node can be a `call_function` or a `call_module`
Returns a Dict that uses Node as key, where value is another Dict that maps from parameter name of that Node to the
size of that parameter
"""


def _analyze_node_size(
    gm: pippy.fx.GraphModule,
) -> Dict[pippy.fx.Node, Dict[str, int]]:
    # state_dict helps us to get parameter sizes
    state_dict = gm.state_dict()

    # Function Parameter Usage
    node_param_sizes: Dict[pippy.fx.Node, Dict[str, int]] = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":  # a parameter node
            param_name = node.target
            assert param_name in state_dict
            param = state_dict[param_name]
            # Find use site of this parameter
            for user in node.users:
                func_param_sizes = node_param_sizes.setdefault(user, {})
                func_param_sizes.setdefault(param_name, param.numel())

    # Module Parameter Usage
    for node in gm.graph.nodes:
        # We calcuate size of a user-defined submodule as a whole
        if node.op == "call_module":
            mod_param_sizes: Dict[str, int] = {}
            submod: torch.nn.Module = gm.get_submodule(node.target)
            for param_name, param in submod.named_parameters():
                mod_param_sizes.setdefault(param_name, param.numel())
            if mod_param_sizes:
                node_param_sizes.setdefault(node, mod_param_sizes)

    for node, param_sizes in node_param_sizes.items():
        logging.debug(f"{node} has params: {param_sizes}")

    return node_param_sizes


"""
Split a model based on a maximum number of parameter and buffer elements a pipeline stage can have
Input:
  mod: `torch.nn.Module` to split
  threshold: maximum number of parameter and buffer elements a stage can have
  max_stages: maximum number of stages; default = -1, no limit
Output:
  A tuple consisting of:
      - a `fx.GraphModule` transformed from the input module with `pipe_split` inserted
      - number of stages the input module is split into
"""


def _split_on_size_threshold_with_max_stages(
    mod: torch.nn.Module, threshold: int, max_stages: int = -1
) -> Tuple[pippy.fx.GraphModule, int]:
    # Trace the user module to get a graph first
    gm: pippy.fx.GraphModule = pippy.fx.symbolic_trace(mod)
    # Analyze size of parameters/buffers used by each node in the graph
    node_param_sizes = _analyze_node_size(gm)

    # Record split positions
    insert_before_nodes: List[pippy.fx.Node] = []

    def new_stage_before(node):
        insert_before_nodes.append(node)

    # Track the parameters we have seen in the current bucket and their total size
    accumulate_size = 0
    accumulate_params: Dict = {}

    for node in gm.graph.nodes:
        if node not in node_param_sizes:
            # The callsite of this node does not involve parameters or buffers
            continue

        # Track the new parameters we see at this node as well as parameters that we have seen in current bucket
        new_size = 0
        new_params: Dict = {}
        repeated_size = 0
        repeated_params: Dict = {}
        param_sizes = node_param_sizes[node]
        if node.op == "call_function":
            # For function, the parameter it uses can be shared with other functions seen previously
            for param_name, size in param_sizes.items():
                if param_name not in accumulate_params:  # new parameter
                    new_params.setdefault(param_name)
                    new_size += size
                else:  # repeated parameter; mark down; use later
                    repeated_params.setdefault(param_name)
                    repeated_size += size
        elif node.op == "call_module":
            # For module, we count its paramters as a single whole
            for param_name, size in param_sizes.items():
                new_size += size

        if (
            accumulate_size + new_size <= threshold
        ):  # can accommodate this node in current bucket
            accumulate_size += new_size
            accumulate_params.update(new_params)
        elif (
            accumulate_size == 0 and new_size > threshold
        ):  # this node becomes a stage
            new_stage_before(node.next)
        else:  # cannot accommodate this node
            new_stage_before(node)
            accumulate_size = repeated_size + new_size
            accumulate_params.clear()
            accumulate_params.update(repeated_params)
            accumulate_params.update(new_params)

    # Insert pipe_split nodes at the recorded positions
    nstages = 1
    for node in insert_before_nodes:
        if nstages == max_stages:
            break
        with gm.graph.inserting_before(node):
            gm.graph.call_function(pipe_split, (), {})
        nstages += 1

    # Since we transformed the graph, we need to recompile the module
    gm.recompile()

    return gm, nstages


"""
Split a model based on a maximum number of parameter and buffer elements a pipeline stage can have
Input:
  mod: `torch.nn.Module` to split
  threshold: maximum number of parameter and buffer elements a stage can have
Output:
  A tuple consisting of:
      - a `fx.GraphModule` transformed from the input module with `pipe_split` inserted
      - number of stages the input module is split into
"""


def split_on_size_threshold(
    mod: torch.nn.Module, threshold: int
) -> Tuple[pippy.fx.GraphModule, int]:
    return _split_on_size_threshold_with_max_stages(mod, threshold)


"""
Split a model into given number of stages, based on equal stage size
Input:
  mod: `torch.nn.Module` to split
  nstages: number of stages to split the module into
Output:
  A tuple consisting of:
      - a `fx.GraphModule` transformed from the input module with `pipe_split` inserted
"""


def split_into_nstages_equal_size(
    mod: torch.nn.Module,
    nstages: int = 1,
) -> pippy.fx.GraphModule:
    param_size = 0
    for param in mod.parameters():
        param_size += param.numel()
    buffer_size = 0
    for buffer in mod.buffers():
        buffer_size += buffer.numel()

    total_size = param_size + buffer_size
    per_stage_size = total_size // nstages
    logging.debug(
        f"Total model size: {total_size}, " f"per stage size: {per_stage_size}"
    )

    gm, rv_nstages = _split_on_size_threshold_with_max_stages(
        mod, per_stage_size, nstages
    )
    assert rv_nstages == nstages

    return gm
