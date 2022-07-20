# Copyright (c) Meta Platforms, Inc. and affiliates
# Adaptation of https://github.com/alpa-projects/alpa/blob/a88992ce3b46024c0a4ee4aa8cb069a62830cec2/alpa/pipeline_parallel/stage_construction.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pippy.fx
import torch
from enum import Enum

from pippy import pipe_split
from pippy.count_flops import compile_model_op_by_op, count_flop_latency_in_mlir_modules_using_llvm

try:
    from numba import njit  # type: ignore
    from numba.typed import List as NumbaList  # type: ignore
    from numba import prange
except ImportError:

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper

    NumbaList = list
    prange = range


class SubmeshSpace(Enum):
    ALL = "ALL"
    POWER_OF_TWO = "POWER_OF_TWO"
    SMALL_POWER_OF_TWO = "SMALL_POWER_OF_TWO"


def get_possible_submesh_shapes(
    n_compute_nodes: int, n_devices_per_node: int, submesh_space: SubmeshSpace
):
    submeshes = []
    i = 1
    while i <= n_devices_per_node:
        submeshes.append((1, i))
        i *= 2
    assert submeshes[-1][1] == n_devices_per_node

    # larger meshes:
    if submesh_space == SubmeshSpace.ALL:
        for i in range(2, n_compute_nodes + 1):
            submeshes.append((i, n_devices_per_node))
    elif submesh_space == SubmeshSpace.POWER_OF_TWO:
        i = 2
        while i <= n_compute_nodes:
            submeshes.append((i, n_devices_per_node))
            i *= 2
    elif submesh_space == SubmeshSpace.SMALL_POWER_OF_TWO:
        i = 2
        while i <= min(n_compute_nodes, 4):
            submeshes.append((i, n_devices_per_node))
            i *= 2
    else:
        raise ValueError(f"Invalid submesh space: {submesh_space}")

    return submeshes


@njit(fastmath=True, nogil=True, cache=True)
def get_optimal_submesh_assignments(
    best_n_stages, F_argmin, n_devices, n_ops, submesh_sizes
):
    """
    Standard backtracking approach to find the optimal op-mesh assignment, starting with
    the optimal number of stages (best_n_stages).

    The return is a list [((layer_start, next_layer_start), submesh_shape_idx, sharding_config_idx)]
    where (layer_start, next_layer_start) is [) slice of the ops and submesh_shape_idx is the submesh
    those ops should be mapped to (sharding_config_idx is currently always 1 but will be eventually used
    pick optimal tensor sharding configuration).
    """
    current_s = best_n_stages
    current_layer = 0
    current_devices = n_devices

    optimal_layer_submesh_assignments = []
    while current_s > 0 and current_layer < n_ops and current_devices > 0:
        next_start_layer, submesh_shape_idx, sharding_config_idx = F_argmin[
            current_s, current_layer, current_devices
        ]
        assert next_start_layer != -1 and current_devices != -1
        optimal_layer_submesh_assignments.append(
            ((current_layer, next_start_layer), submesh_shape_idx, sharding_config_idx)
        )
        current_s -= 1
        current_layer = next_start_layer
        current_devices -= submesh_sizes[submesh_shape_idx]

    assert current_s == 0 and current_layer == n_ops and current_devices == 0

    return optimal_layer_submesh_assignments


@njit(fastmath=True, parallel=True, nogil=True, cache=True)
def inter_op_dp_inner_loop(
    n_layers, n_devices, submesh_sizes, valid_idxs_costs, max_n_succ_stages
):
    """
    Equation 3 from the Alpa paper. Primary difference from the paper is the
    s - 1 <= max_n_succ_stages check, which is used to characterize memory capacity
    of each stage placement (if s - 1 > max_n_succ_stages check then placing that stage
    would lead to OOM and thus continue).
    """
    F = np.full((n_layers + 1, n_layers + 1, n_devices + 1), np.inf, dtype=np.float32)
    F_stage_max = np.full(
        (n_layers + 1, n_layers + 1, n_devices + 1), 0.0, dtype=np.float32
    )
    F_argmin = np.full(
        (n_layers + 1, n_layers + 1, n_devices + 1, 3), -1, dtype=np.int32
    )
    F[0, n_layers, 0] = 0

    for d in prange(1, n_devices + 1):
        for (
            l,
            i,
            submesh_shape_idx,
            sharding_config_idx,
            stage_cost,
        ) in valid_idxs_costs:
            l, i, submesh_shape_idx, sharding_config_idx = map(
                int, (l, i, submesh_shape_idx, sharding_config_idx)
            )

            n_submesh_devices = submesh_sizes[submesh_shape_idx]
            if n_submesh_devices <= d:
                for s in prange(1, n_layers + 1):
                    if (
                        s - 1
                        > max_n_succ_stages[
                            l, i, submesh_shape_idx, sharding_config_idx
                        ]
                    ):
                        continue

                    new_cost = F[s - 1, i + 1, d - n_submesh_devices] + stage_cost
                    if new_cost < F[s, l, d]:
                        F[s, l, d] = new_cost
                        F_argmin[s, l, d] = (
                            i + 1,
                            submesh_shape_idx,
                            sharding_config_idx,
                        )
                        F_stage_max[s, l, d] = max(
                            F_stage_max[s - 1, i + 1, d - n_submesh_devices], stage_cost
                        )

    return F, F_stage_max, F_argmin


def inter_op_dp(
    n_layers: int,
    n_devices: int,
    n_microbatches: int,
    submesh_shapes: List[Tuple[int, int]],
    intra_compute_costs,
    max_n_succ_stages,
):
    """
    DP to compute optimal latency and number of pipeline stages and mapping of
    stages to compute cluster submeshes.
    """
    min_cost = np.inf
    best_solution = None
    prev_intra_cost = 0.0
    gap = 1e-6

    submesh_sizes: list = NumbaList()
    for n, m in submesh_shapes:
        submesh_sizes.append(n * m)

    overall_max_layer = 0
    for loop_i, intra_cost in enumerate(np.sort(np.unique(intra_compute_costs))):
        if intra_cost - prev_intra_cost < gap:
            continue
        if intra_cost * n_microbatches >= min_cost:
            break

        # Optimization that lifts a check for stage_cost <= t_max_stage_cost
        # out of the inner dp loop (see alpa/~/stage_construction.py#L121).
        # This yields a ~100-200x improvement over the baseline implementation.
        valid_cost_idxs = np.transpose((intra_compute_costs <= intra_cost).nonzero())
        # This corresponds to the i of k <= i <= K from eqn. 3 in the alpa paper.
        valid_cost_idxs = valid_cost_idxs[
            valid_cost_idxs[:, 0] <= valid_cost_idxs[:, 1]
        ]
        valid_costs = intra_compute_costs[tuple(valid_cost_idxs.T)]
        valid_idxs_costs = np.hstack([valid_cost_idxs, valid_costs[:, np.newaxis]])
        # sort by descending layer idx because DP initializes F[0, n_layers, 0] = 0
        valid_idxs_costs = valid_idxs_costs[np.flip(valid_cost_idxs[:, 1].argsort())]
        max_layer = int(valid_idxs_costs[:, 1][0])
        overall_max_layer = max(overall_max_layer, max_layer)

        logging.info(f"DP outer loop {loop_i}")
        F, F_stage_max, F_argmin = inter_op_dp_inner_loop(
            max_layer + 1, n_devices, submesh_sizes, valid_idxs_costs, max_n_succ_stages
        )

        best_n_stages = F[:, 0, n_devices].argmin()
        all_stages_cost = F[best_n_stages, 0, n_devices]
        slowest_stage_cost = F_stage_max[best_n_stages, 0, n_devices]
        logging.info(f"DP {all_stages_cost=} {slowest_stage_cost=}")
        if np.isinf(all_stages_cost):
            continue
        slowest_stage_total_cost = (n_microbatches - 1) * slowest_stage_cost

        if all_stages_cost + slowest_stage_total_cost < min_cost:
            min_cost = all_stages_cost + slowest_stage_total_cost
            best_solution = best_n_stages, F_argmin
            logging.info(f"DP updating best_n_stages {best_n_stages}")
        prev_intra_cost = intra_cost

    assert best_solution is not None
    best_n_stages, F_argmin = best_solution
    optimal_layer_submesh_assignments = get_optimal_submesh_assignments(
        best_n_stages, F_argmin, n_devices, overall_max_layer + 1, submesh_sizes
    )
    optimal_layer_submesh_assignments = [
        ((a, b), submesh_shapes[c], d)
        for (a, b), c, d in optimal_layer_submesh_assignments
    ]
    return optimal_layer_submesh_assignments


@dataclass
class AutoParallelConfig:
    n_compute_nodes: int
    n_devices_per_node: int
    n_microbatches: int
    n_autosharding_configs: int = 1
    submesh_space: SubmeshSpace = SubmeshSpace.ALL


def estimate_intra_costs(
    model,
    traced,
    example_inputs,
    submesh_shapes,
    tracer,
    tracer_kwargs,
    max_n_succ_stages=4096,
    n_autosharding_configs=1,
):
    n_graph_nodes = len(traced.graph.nodes)
    n_submesh_choices = len(submesh_shapes)

    # Lower op-by-op to MLIR modules (matched to op name).
    logging.info("Compiling model through MLIR")
    node_mlir_modules, node_shapes = compile_model_op_by_op(
        model, example_inputs, tracer=tracer, return_shapes=True, **tracer_kwargs
    )
    # Count up all of the floating point ops and multiply by
    # op latency (TODO: arithmetic intensity model).
    logging.info("Counting FP latencies for torch ops")
    known_latencies = count_flop_latency_in_mlir_modules_using_llvm(node_mlir_modules)
    all_latencies = np.zeros(n_graph_nodes)
    cum_inputs = []
    nodes = []
    for i, node in enumerate(traced.graph.nodes):
        all_latencies[i] = known_latencies.get(node.name, 0)
        node: torch.fx.Node
        nodes.append(node)
        if cum_inputs:
            cum_inputs.append(
                set((n.name, node.name) for n in node.all_input_nodes) | cum_inputs[-1]
            )
        else:
            cum_inputs.append(set((n.name, node.name) for n in node.all_input_nodes))

    cum_outputs = []
    for node in reversed(traced.graph.nodes):
        if cum_outputs:
            cum_outputs.append(
                set((node.name, n.name) for n in node.users) | cum_outputs[-1]
            )
        else:
            cum_outputs.append(set((node.name, n.name) for n in node.users))

    cum_outputs = cum_outputs[::-1]

    ingress_traffic = np.zeros((n_graph_nodes, n_graph_nodes))
    egress_traffic = np.zeros((n_graph_nodes, n_graph_nodes))
    for i in range(n_graph_nodes):
        for j in range(i, n_graph_nodes):
            input_edges = (cum_inputs[j] - cum_inputs[i]) | set(
                [(n.name, nodes[i].name) for n in nodes[i].all_input_nodes]
            )
            output_edges = (cum_outputs[i] - cum_outputs[j]) | set(
                [(nodes[j].name, n.name) for n in nodes[j].users]
            )
            ingress_edges = input_edges - output_edges
            egress_edges = output_edges - input_edges
            ingress_traffic[i, j] = sum(
                np.prod(node_shapes.get(n, [0])[0]) for n, _ in ingress_edges
            )
            egress_traffic[i, j] = sum(
                np.prod(node_shapes.get(n, [0])[0]) for n, _ in egress_edges
            )

    # intra_costs[i, j, k, l] is the total latency of [node_i, ..., node_j]
    # for the k submesh choice and lth autosharding choice (with inclusive endpoints).
    # Currently we only have one submesh choice (all of the devices) and one autosharding choices (none).
    intra_costs = np.zeros(
        (n_graph_nodes, n_graph_nodes, n_submesh_choices, n_autosharding_configs)
    )
    # Prefix sums such that jth prefix - ith prefix + op latency = latency from ith to jth op.
    cum_latencies = np.cumsum(all_latencies)
    for i in range(n_graph_nodes):
        for j in range(i, n_graph_nodes):
            lat = cum_latencies[j] - cum_latencies[i] + all_latencies[i]
            for k, shape in enumerate(submesh_shapes):
                intra_costs[i, j, k, :] = (
                    (lat / np.prod(shape))
                    + ingress_traffic[i, j]
                    + egress_traffic[i, j]
                )

    # TODO: obviously this isn't right.
    max_n_succ_stages = np.full(
        (n_graph_nodes, n_graph_nodes, n_submesh_choices, n_autosharding_configs),
        max_n_succ_stages,
    )

    return intra_costs, max_n_succ_stages


def dp_auto_parallel(config: AutoParallelConfig):
    def _dp_auto_parallel(
        mod: torch.nn.Module,
        fx_mod: pippy.fx.GraphModule,
        example_inputs,
        tracer,
        tracer_kwargs,
    ):
        submesh_shapes = get_possible_submesh_shapes(
            n_compute_nodes=config.n_compute_nodes,
            n_devices_per_node=config.n_devices_per_node,
            submesh_space=config.submesh_space,
        )
        logging.info("Estimating intra-op latencies")
        intra_costs, max_n_succ_stages = estimate_intra_costs(
            mod,
            fx_mod,
            example_inputs,
            submesh_shapes,
            tracer=tracer,
            tracer_kwargs=tracer_kwargs,
            n_autosharding_configs=config.n_autosharding_configs,
        )
        _intra_costs = intra_costs.squeeze()
        n_graph_nodes = len(fx_mod.graph.nodes)
        logging.info("Computing optimal split using DP")
        optimal_layer_submesh_assignments = inter_op_dp(
            n_layers=n_graph_nodes,
            n_devices=config.n_compute_nodes * config.n_devices_per_node,
            n_microbatches=config.n_microbatches,
            submesh_shapes=submesh_shapes,
            intra_compute_costs=intra_costs,
            max_n_succ_stages=max_n_succ_stages,
        )
        logging.info(
            f"DP optimal layer submesh assignments {optimal_layer_submesh_assignments}"
        )
        split_points = {
            current_layer
            for (
                (current_layer, _next_start_layer),
                _submesh_choice,
                _autosharding_choice,
            ) in optimal_layer_submesh_assignments
        }
        for i, node in reversed(list(enumerate(fx_mod.graph.nodes))):
            if i in split_points:
                with fx_mod.graph.inserting_before(node):
                    fx_mod.graph.call_function(pipe_split, (), {})
        fx_mod.recompile()
        return fx_mod

    return _dp_auto_parallel
