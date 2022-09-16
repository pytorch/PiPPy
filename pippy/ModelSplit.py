# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from typing import Dict

import torch
import pippy.fx
from pippy.IR import pipe_split

def analyze_node_sizes(gm: pippy.fx.GraphModule):
    state_dict = gm.state_dict()
    print(f'\n======= Original Module Traced =======')
    print(gm)

    print(f'\n======= Graph tabular =======')
    gm.graph.print_tabular()

    print('\n======= Function Parameter Usage =======')
    node_param_sizes : Dict = {}
    for node in gm.graph.nodes:
        if node.op == 'get_attr':
            param_name = node.target
            assert param_name in state_dict
            param = state_dict[param_name]
            for user in node.users:
                param_sizes = node_param_sizes.setdefault(user, {})
                param_sizes.setdefault(param_name, param.numel())

    for node, param_sizes in node_param_sizes.items():
        print(f'{node} has params: {param_sizes}')

    """
    print('\n======= Functions under Graph =======')
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            param_sizes : Dict = {}
            for arg_node in node.args:
                if isinstance(arg_node, pippy.fx.node.Node) and arg_node.op == 'get_attr':
                    param_name = arg_node.target
                    assert param_name in state_dict
                    param = state_dict[param_name]
                    param_sizes.setdefault(param_name, param.numel())
            if param_sizes:
                print(f'{node} has params: {param_sizes}')
    """

    print('\n======= Module Parameter Usage =======')
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            param_sizes : Dict = {}
            submod: torch.nn.Module = gm.get_submodule(node.target)
            for param_name, param in submod.named_parameters():
                param_sizes.setdefault(param_name, param.numel())
            if param_sizes:
                node_param_sizes.setdefault(node, param_sizes)
                print(f'{node} has params: {param_sizes}')

    return node_param_sizes

def auto_split_based_on_size_threshold(mod : torch.nn.Module, threshold):
    gm : pippy.fx.GraphModule = pippy.fx.symbolic_trace(mod)
    node_param_sizes = analyze_node_sizes(gm)

    insert_before_nodes = []

    def new_stage_before(node):
        insert_before_nodes.append(node)

    accumulate_size = 0
    accumulate_params = {}

    for node in gm.graph.nodes:
        if node not in node_param_sizes:
            continue
        new_size = 0
        new_params = {}
        repeated_size = 0
        repeated_params = {}
        param_sizes = node_param_sizes[node]
        if (node.op == 'call_function'):
            for param_name, size in param_sizes.items():
                if param_name not in accumulate_params:
                    new_params.setdefault(param_name)
                    new_size += size
                else: # repeated parameter; mark down
                    repeated_params.setdefault(param_name)
                    repeated_size += size
        elif (node.op == 'call_module'):
            for param_name, size in param_sizes.items():
                new_size += size

        if accumulate_size + new_size <= threshold: # can accommodate this node
            accumulate_size += new_size
            accumulate_params.update(new_params)
        elif accumulate_size == 0 and new_size > threshold: # this node becomes a stage
            new_stage_before(node.next)
        else:   # cannot accommodate this node
            new_stage_before(node)
            accumulate_size = repeated_size + new_size
            accumulate_params.clear()
            accumulate_params.update(repeated_params)
            accumulate_params.update(new_params)

    for node in insert_before_nodes:
        with gm.graph.inserting_before(node):
            gm.graph.call_function(pipe_split, (), {})

    gm.recompile()
    
    return gm
