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
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

from torch import fx

from pippy import pipe_split

try:
    from numba import njit  # type: ignore
    from numba.typed import List as NumbaList  # type: ignore
except ImportError:

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper

    NumbaList = list


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


NUMPY_RANDOM_SEED = 42


def estimate_intra_costs(
    n_submesh_choices,
    n_layers,
    max_n_succ_stages=4096,
    n_autosharding_configs=1,
):
    np.random.seed(NUMPY_RANDOM_SEED)
    intra_costs = np.random.rand(
        n_layers, n_layers, n_submesh_choices, n_autosharding_configs
    )
    max_n_succ_stages = np.full(
        (n_layers, n_layers, n_submesh_choices, n_autosharding_configs),
        max_n_succ_stages,
    )
    return intra_costs, max_n_succ_stages


@njit(fastmath=True)
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
            (
                (current_layer, next_start_layer),
                submesh_shape_idx,
                sharding_config_idx,
            )
        )
        current_s -= 1
        current_layer = next_start_layer
        current_devices -= submesh_sizes[submesh_shape_idx]

    assert current_s == 0 and current_layer == n_ops and current_devices == 0

    return optimal_layer_submesh_assignments


@njit(fastmath=True)
def inter_op_dp_inner_loop(
    n_layers, n_devices, submesh_sizes, valid_idxs_costs, max_n_succ_stages
):
    """
    Equation 3 from the Alpa paper. Primary difference from the paper is the
    s - 1 <= max_n_succ_stages check, which is used to characterize memory capacity
    of each stage placement (if s - 1 > max_n_succ_stages check then placing that stage
    would lead to OOM and thus continue).
    """
    F = np.full(
        (n_layers + 1, n_layers + 1, n_devices + 1), np.inf, dtype=np.float32
    )
    F_stage_max = np.full(
        (n_layers + 1, n_layers + 1, n_devices + 1), 0.0, dtype=np.float32
    )
    F_argmin = np.full(
        (n_layers + 1, n_layers + 1, n_devices + 1, 3), -1, dtype=np.int32
    )
    F[0, n_layers, 0] = 0

    for d in range(1, n_devices + 1):
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
                for s in range(1, n_layers + 1):
                    if (
                        s - 1
                        > max_n_succ_stages[
                            l, i, submesh_shape_idx, sharding_config_idx
                        ]
                    ):
                        continue

                    new_cost = (
                        F[s - 1, i + 1, d - n_submesh_devices] + stage_cost
                    )
                    if new_cost < F[s, l, d]:
                        F[s, l, d] = new_cost
                        F_argmin[s, l, d] = (
                            i + 1,
                            submesh_shape_idx,
                            sharding_config_idx,
                        )
                        F_stage_max[s, l, d] = max(
                            F_stage_max[s - 1, i + 1, d - n_submesh_devices],
                            stage_cost,
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

    for intra_cost in np.sort(np.unique(intra_compute_costs)):
        if intra_cost - prev_intra_cost < gap:
            continue
        if intra_cost * n_microbatches >= min_cost:
            break

        # Optimization that lifts a check for stage_cost <= t_max_stage_cost
        # out of the inner dp loop (see alpa/~/stage_construction.py#L121).
        # This yields a ~100-200x improvement over the baseline implementation.
        valid_cost_idxs = np.transpose(
            (intra_compute_costs <= intra_cost).nonzero()
        )
        # This corresponds to the i of k <= i <= K from eqn. 3 in the alpa paper.
        valid_cost_idxs = valid_cost_idxs[
            valid_cost_idxs[:, 0] <= valid_cost_idxs[:, 1]
        ]
        valid_costs = intra_compute_costs[tuple(valid_cost_idxs.T)]
        valid_idxs_costs = np.hstack(
            [valid_cost_idxs, valid_costs[:, np.newaxis]]
        )

        F, F_stage_max, F_argmin = inter_op_dp_inner_loop(
            n_layers,
            n_devices,
            submesh_sizes,
            valid_idxs_costs,
            max_n_succ_stages,
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
        prev_intra_cost = intra_cost

    assert best_solution is not None
    best_n_stages, F_argmin = best_solution
    optimal_layer_submesh_assignments = get_optimal_submesh_assignments(
        best_n_stages, F_argmin, n_devices, n_layers, submesh_sizes
    )
    return optimal_layer_submesh_assignments


@dataclass
class AutoParallelConfig:
    n_compute_nodes: int
    n_devices_per_node: int
    n_microbatches: int
    submesh_space: SubmeshSpace = SubmeshSpace.ALL


def dp_auto_parallel(config: AutoParallelConfig):
    def _dp_auto_parallel(fx_mod: fx.GraphModule):
        n_graph_nodes = len(fx_mod.graph.nodes)
        submesh_shapes = get_possible_submesh_shapes(
            n_compute_nodes=config.n_compute_nodes,
            n_devices_per_node=config.n_devices_per_node,
            submesh_space=config.submesh_space,
        )
        intra_costs, max_n_succ_stages = estimate_intra_costs(
            len(submesh_shapes), n_layers=n_graph_nodes
        )
        optimal_layer_submesh_assignments = inter_op_dp(
            n_layers=n_graph_nodes,
            n_devices=config.n_compute_nodes * config.n_devices_per_node,
            n_microbatches=config.n_microbatches,
            submesh_shapes=submesh_shapes,
            intra_compute_costs=intra_costs,
            max_n_succ_stages=max_n_succ_stages,
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
