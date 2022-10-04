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
from typing import Callable, Any

import numpy as np
import torch
from enum import Enum

from pippy import pipe_split

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
) -> tuple[tuple[int, int]]:
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

    return tuple(submeshes)


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
    n_devices: int,
    n_microbatches: int,
    submesh_shapes: tuple[tuple[int, int]],
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
    num_layers = len(intra_compute_costs)

    submesh_sizes: list = NumbaList()
    for n, m in submesh_shapes:
        submesh_sizes.append(n * m)

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
        if len(valid_cost_idxs) == 0:
            continue
        valid_costs = intra_compute_costs[tuple(valid_cost_idxs.T)]
        valid_idxs_costs = np.hstack([valid_cost_idxs, valid_costs[:, np.newaxis]])
        # sort by descending layer idx because DP initializes F[0, n_layers, 0] = 0
        valid_idxs_costs = valid_idxs_costs[np.flip(valid_cost_idxs[:, 1].argsort())]

        logging.info(f"DP outer loop {loop_i}")
        F, F_stage_max, F_argmin = inter_op_dp_inner_loop(
            num_layers, n_devices, submesh_sizes, valid_idxs_costs, max_n_succ_stages
        )

        best_n_stages = F[:, 0, n_devices].argmin()
        all_stages_cost = F[best_n_stages, 0, n_devices]
        slowest_stage_cost = F_stage_max[best_n_stages, 0, n_devices]
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
        best_n_stages, F_argmin, n_devices, num_layers, submesh_sizes
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
    intra_op_costs_estimator: Callable[
        [
            torch.nn.Module,
            torch.fx.GraphModule,
            # example inputs,
            tuple[torch.Tensor, ...],
            # submesh shapes,
            tuple[tuple[int, int], ...],
            torch.fx.Tracer,
            # tracer kwargs,
            dict[str, Any],
        ],
        tuple[np.ndarray, np.ndarray, tuple[tuple[str, int]]],
    ]
    n_autosharding_configs: int = 1
    submesh_space: SubmeshSpace = SubmeshSpace.ALL


def dp_auto_parallel(config: AutoParallelConfig):
    def _dp_auto_parallel(
        mod: torch.nn.Module,
        fx_mod: torch.fx.GraphModule,
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
        (
            intra_costs,
            max_n_succ_stages,
            node_name_graph_idx,
        ) = config.intra_op_costs_estimator(
            mod,
            fx_mod,
            example_inputs,
            submesh_shapes,
            tracer,
            tracer_kwargs,
        )
        logging.info("Computing optimal split using DP")

        # optimal_layer_submesh_assignments is a tuple of
        # ([ith node, jth node), (num_hosts_i, num_devices_i), autosharding_config)
        # where (num_hosts_i, num_devices_i) describes the ith submesh slice and
        # such that Σi (num_hosts_i, num_devices_i) = total_num_hosts * devices_per_host
        # (i.e., total # of devices). Note that mapping submeshes to physical devices can (and should)
        # be specified by the user (in the alpa paper they use a greedy heuristic - devices are assigned to
        # larger submeshes first and then to smaller ones and ties are broken by physically clustering
        # pipeline stages.
        optimal_layer_submesh_assignments = inter_op_dp(
            n_devices=config.n_compute_nodes * config.n_devices_per_node,
            n_microbatches=config.n_microbatches,
            submesh_shapes=submesh_shapes,
            intra_compute_costs=intra_costs,
            max_n_succ_stages=max_n_succ_stages,
        )
        logging.info(
            f"DP optimal layer submesh assignments {optimal_layer_submesh_assignments}"
        )
        # node_name_graph_idx is a tuple (node_name, j) where j is in the index in the
        # original toposort order of the fx graph. The reason for this level of indirection
        # (i.e., ith entry in node_name_graph_idx actually maps to jth in graph is because not all
        # nodes have valid latencies (e.g., inputholder) and thus the DP does not take those into consideration
        split_points = [
            node_name_graph_idx[next_start_layer][1]
            for (
                (_current_layer, next_start_layer),
                _submesh_choice,
                _autosharding_choice,
            ) in optimal_layer_submesh_assignments[:-1]
        ]
        nodes = list(enumerate(fx_mod.graph.nodes))
        for i, node in reversed(nodes):
            if i in split_points:
                nodes.insert(i, (i, "split"))
                with fx_mod.graph.inserting_before(node):
                    fx_mod.graph.call_function(pipe_split, (), {})
        fx_mod.recompile()
        return fx_mod

    return _dp_auto_parallel
