# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Callable, Dict, List, Tuple

import torch
import torch.fx as fx

from ._IR import aten_pipe_split_alias


logger = logging.getLogger(__name__)

"""
Analyze size of parameters/buffers used by each node in the graph
Here node can be a `call_function` or a `call_module`
Returns a Dict that uses Node as key, where value is another Dict that maps from parameter name of that Node to the
size of that parameter
"""


def _analyze_node_size(
    gm: fx.GraphModule,
) -> Dict[fx.Node, Dict[str, int]]:
    # state_dict helps us to get parameter sizes
    state_dict = gm.state_dict()

    # Function Parameter Usage
    node_param_sizes: Dict[fx.Node, Dict[str, int]] = {}
    for node in gm.graph.nodes:
        if node.op == "get_attr":  # a parameter node
            param_name = node.target
            if param_name not in state_dict:
                # In some cases, attr node is not a parameter or buffer, we just skip it
                continue
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
        logger.debug(f"{node} has params: {param_sizes}")

    return node_param_sizes


"""
Split a model based on a maximum number of parameter and buffer elements a pipeline stage can have
Input:
  gm: `fx.GraphModule` to split
  threshold: maximum number of parameter and buffer elements a stage can have
  max_stages: maximum number of stages; default = -1, no limit
Output:
  A tuple consisting of:
      - a `fx.GraphModule` transformed from the input module with `pipe_split` inserted
      - number of stages the input module is split into
"""


def _split_on_size_threshold_with_max_stages(
    gm: fx.GraphModule,
    threshold: int,
    max_stages: int = -1,
) -> Tuple[fx.GraphModule, int]:
    # Analyze size of parameters/buffers used by each node in the graph
    node_param_sizes = _analyze_node_size(gm)

    # Record split positions
    insert_before_nodes: List[fx.Node] = []

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
            gm.graph.call_function(aten_pipe_split_alias, (), {})
        nstages += 1

    # Since we transformed the graph, we need to recompile the module
    gm.recompile()

    return gm, nstages


"""
Create a Callable that splits a module based on a maximum number of parameter and buffer elements a pipeline stage can
have.
Input:
  threshold: maximum number of parameter and buffer elements a stage can have
Output:
  a Callable that transforms an input `fx.GraphModule` into an output `fx.GraphModule` that has `pipe_split` inserted
  before a stage reaches the `threshold` size
"""


def split_on_size_threshold(
    threshold: int,
) -> Callable[[fx.GraphModule], fx.GraphModule]:
    def _split_on_size_threshold(
        gm: fx.GraphModule,
    ) -> fx.GraphModule:
        gm, _ = _split_on_size_threshold_with_max_stages(gm, threshold)
        return gm

    return _split_on_size_threshold


"""
Create a Callable that splits a model into given number of stages, based on equal stage size
Input:
  nstages: number of stages to split the module into
Output:
  a Callable that transforms an input `fx.GraphModule` into an output `fx.GraphModule` that has `pipe_split` inserted
  between `nstages` stages
"""


def split_into_equal_size(
    nstages: int = 1,
) -> Callable[[fx.GraphModule], fx.GraphModule]:
    def _split_into_nstages_equal_size(
        gm: fx.GraphModule,
    ) -> fx.GraphModule:
        param_size = 0
        for param in gm.parameters():
            param_size += param.numel()
        buffer_size = 0
        for buffer in gm.buffers():
            buffer_size += buffer.numel()

        total_size = param_size + buffer_size
        per_stage_size = total_size // nstages
        logger.debug(
            f"Total model size: {total_size}, "
            f"per stage size: {per_stage_size}"
        )

        gm, rv_nstages = _split_on_size_threshold_with_max_stages(
            gm, per_stage_size, nstages
        )
        assert rv_nstages == nstages
        return gm

    return _split_into_nstages_equal_size


"""
Create a Callable that splits a model into a given number of stages, based on the computation graph, while
trying to minimize the communication between the stages and to balance the computation
Input:
  nstages: the number of stages to split the module into
Output:
  a Callable that transforms an input `fx.GraphModule` into an output `fx.GraphModule` that has `pipe_split` inserted
  between `nstages` stages
"""


def split_by_graph(nstages: int) -> Callable[[fx.GraphModule], fx.GraphModule]:
    def _split_by_graph(
        gm: fx.GraphModule,
    ) -> fx.GraphModule:
        node_param_sizes = _analyze_node_size(gm)
        node2stage = split_by_graph_with_num_stages(
            gm, nstages, node_param_sizes
        )

        # Remove existing split points
        for node in gm.graph.nodes:
            if "pipe_split" in node.name:
                gm.graph.erase_node(node)

        # Modify the graph by grouping nodes on the same stage and adding
        # pipe_splits between the stages
        node_order = [node for node in gm.graph.nodes if node in node2stage]
        last_node = None
        for stage_idx in range(nstages):
            nodes_at_stage = [
                node
                for node in node_order
                if node in node2stage and node2stage[node] == stage_idx
            ]
            for idx, node in enumerate(nodes_at_stage):
                if last_node is not None and last_node.next != node:
                    last_node.append(node)
                last_node = node
            # Insert pipe_split nodes after each stage, except the last one
            if stage_idx + 1 != nstages and last_node is not None:
                with gm.graph.inserting_after(last_node):
                    last_node = gm.graph.call_function(
                        aten_pipe_split_alias, (), {}
                    )

        # Since we transformed the graph, recompile the module
        gm.recompile()
        gm.graph.lint()
        return gm

    return _split_by_graph
