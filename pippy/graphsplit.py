# Copyright (c) Meta Platforms, Inc. and affiliates
import logging

import time

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.fx as fx
from torch import ones, zeros

from ._debug import PIPPY_VERBOSITY

_scipy_is_available = False
try:
    from scipy.optimize import Bounds, LinearConstraint, milp

    _scipy_is_available = True
except ImportError:
    _scipy_is_available = False


logger = logging.getLogger(__name__)


@dataclass
class Node:
    name: str
    weight: int
    stage: Optional[int]
    gm_node: fx.Node


@dataclass
class Edge:
    source: int
    target: int
    weight: int


MAX_MEMORY_IMBALANCE = 2.0
MAX_COMMUNICATION_IMBALANCE = 1.05
SCIPY_TIME_LIMIT_SEC = 60


"""
Splits a model into a given number of stages, based on the computation graph, while trying to
minimize the communication between the stages and to balance the computation across stages. The
optimization is done via solving a mixed-integer linear program (MILP) using `scipy`.
Input:
  gm: `fx.GraphModule` to split
  num_stages: the number of stages to split the module into
  node_param_sizes: a Dict that uses `fx.Node` as the key and another Dict mapping the parameter name to the
                    size of that parameter as the value
Output:
  a Dict with `fx.Node` as the key and the stage of the node as te value
Raises:
  RuntimeError: If `scipy` is not available
"""


def split_by_graph_with_num_stages(
    gm: fx.GraphModule,
    num_stages: int,
    node_param_sizes: Dict[fx.Node, Dict[str, int]],
) -> Dict[fx.Node, int]:
    if not _scipy_is_available:
        raise RuntimeError(
            "Please install scipy 1.9.0+ to use `split_by_graph`. This is done "
            "using `pip install scipy`."
        )

    # Extract the graph data
    nodes, edges = _build_splitting_graph(gm, node_param_sizes)

    # Run the splitting algorithm with the specified options
    _split_by_milp(
        nodes,
        edges,
        num_stages,
        MAX_MEMORY_IMBALANCE,
        MAX_COMMUNICATION_IMBALANCE,
    )

    # Print the resulting stats
    if PIPPY_VERBOSITY == "DEBUG":
        _print_splitting_stats(nodes, edges, num_stages)

    return {n.gm_node: n.stage for n in nodes if n.stage is not None}


"""
Extract the weighted computation graph for pipeline splitting. The weights of nodes (operations) represent
the memory footprint of parameters/buffers used by the ops. The weights of data flow edges correspond to
the communication between the source and the target ops, that is, the size of the output of the source.
Input:
  gm: `fx.GraphModule` to split
  node_param_sizes: a Dict that uses `fx.Node` as the key and another Dict mapping the parameter name to the
                    size of that parameter as the value
Output:
  a list of nodes with a list of edges
"""


def _build_splitting_graph(
    gm: fx.GraphModule,
    node_param_sizes: Dict[fx.Node, Dict[str, int]],
) -> Tuple[List[Node], List[Edge]]:
    # Build nodes
    nodes: List[Node] = []
    activation_size = {}
    node_index = {}
    for node in gm.graph.nodes:
        if node.op == "output" or "pipe_split" in node.name:
            continue
        if node in node_param_sizes:
            weight = sum(v for _, v in node_param_sizes[node].items())
        else:
            weight = 0
        if "example_value" in node.meta:
            tensors = node.meta["example_value"]
            if isinstance(tensors, torch.Tensor):
                tensors = [tensors]
            activation_size[node] = sum(t.numel() for t in tensors)
        else:
            activation_size[node] = 0
        node_index[node.name] = len(nodes)
        nodes.append(Node(node.name, weight, None, node))

    # Build edges
    edges: List[Edge] = []
    for node in gm.graph.nodes:
        if node.op == "output" or "pipe_split" in node.name:
            continue
        for pred in node.args:
            if not isinstance(pred, torch.fx.node.Node):
                continue
            source_idx = node_index[pred.name]
            target_idx = node_index[node.name]
            weight = activation_size[pred]
            edges.append(Edge(source_idx, target_idx, weight))

    # Verify the collected data
    assert (
        sum(node.weight for node in nodes) > 0
    ), "node weights cannot be empty"
    assert (
        sum(edge.weight for edge in edges) > 0
    ), "edge weights cannot be empty"

    return nodes, edges


"""
Construct and solve a MILP for splitting the computation graph into a specified number of stages.
Input:
  nodes: the list of weighted nodes in the computation graph
  edges: the list of weighted edges in the computation graph
  num_stages: the number of stages to split the graph into
  allowed_node_imbalance: the maximum allowed node (memory) imbalance across the stages
  allowed_edge_imbalance: the maximum allowed edge (communication) imbalance across the stages
Raises:
  ValueError: If the constructed MILP is not feasible or cannot be solved with `scipy`
"""


def _split_by_milp(
    nodes: List[Node],
    edges: List[Edge],
    num_stages: int,
    allowed_node_imbalance: float,
    allowed_edge_imbalance: float,
):
    # Assume we have a graph with N nodes, M edges; the goal is to split it
    # into K node-disjoint stages (clusters of nodes)
    N = len(nodes)
    M = len(edges)
    K = num_stages
    logger.info(
        "Splitting graph with {} nodes and {} edges into {} stages".format(
            N, M, K
        )
    )
    assert allowed_node_imbalance >= 1.0 and allowed_edge_imbalance >= 1.0

    # The optimization model contains N * K binary node variables such that
    # x[i, j] = 1 if i-th node belongs to j-th stage, and x[i, j] = 0 otherwise.
    num_node_vars = N * K
    # Similarly, M * K binary edge variables x[i, j] = 1 if and only if both
    # endpoints of the i-th edge belong to the j-th stage.
    num_edge_vars = M * K
    # An extra auxiliary variable for optimization.
    num_aux_vars = 1
    num_vars = num_node_vars + num_edge_vars + num_aux_vars

    def node_var(node_idx: int, stage: int) -> int:
        return node_idx * K + stage

    def edge_var(edge_idx: int, stage: int) -> int:
        return num_node_vars + edge_idx * K + stage

    edge_aux_var = num_node_vars + num_edge_vars

    # Define constraints for the optimization.
    constraints = []

    # Constraint 1:
    #   - node/edge variables are synchronized with each other;
    for i in range(M):
        edge = edges[i]
        for j in range(K):
            # x[edge_var(i, j)] <= x[node_var(edge.source, j)]
            A1 = zeros(num_vars)
            A1[edge_var(i, j)] = 1
            A1[node_var(edge.source, j)] = -1
            constraints.append(LinearConstraint(A=A1, ub=0))
            # x[edge_var(i, j)] <= x[node_var(edge.target, j)]
            A2 = zeros(num_vars)
            A2[edge_var(i, j)] = 1
            A2[node_var(edge.target, j)] = -1
            constraints.append(LinearConstraint(A=A2, ub=0))
            # x[node_var(edge.source, j)] + x[node_var(edge.target, j)] - 1 <= x[edge_var(i, j)]
            A3 = zeros(num_vars)
            A3[node_var(edge.source, j)] = 1
            A3[node_var(edge.target, j)] = 1
            A3[edge_var(i, j)] = -1
            constraints.append(LinearConstraint(A=A3, ub=1))

    # Constraint 2:
    #   - every node belongs to some stage;
    for i in range(N):
        A = zeros(num_vars)
        for j in range(K):
            A[node_var(i, j)] = 1
        constraints.append(LinearConstraint(A=A, lb=1, ub=1))

    # Constraint 3:
    #   - every stage contains at least one edge;
    for j in range(K):
        A = zeros(num_vars)
        for i in range(M):
            A[edge_var(i, j)] = 1
        constraints.append(LinearConstraint(A=A, lb=1))

    # Constraint 4:
    #   - edges go from a lower-index stage to an upper-index stage;
    multiplier = [2 ** (K - j - 1) for j in range(K)]
    for i in range(M):
        edge = edges[i]
        A = zeros(num_vars)
        for j in range(K):
            A[node_var(edge.target, j)] = multiplier[j]
            A[node_var(edge.source, j)] = -multiplier[j]
        constraints.append(LinearConstraint(A=A, ub=0))

    # Constraint 5:
    #   - nodes in every stage have (approximately) the same total weight;
    sum_node_weights = sum(node.weight for node in nodes)
    max_node_weight_per_stage = (
        sum_node_weights * allowed_node_imbalance / float(K)
    )
    for j in range(K):
        A = zeros(num_vars)
        for i in range(N):
            A[node_var(i, j)] = nodes[i].weight
        constraints.append(LinearConstraint(A=A, ub=max_node_weight_per_stage))

    # Constraint 6:
    #   - edges in every stage have (approximately) the same total weight;
    sum_edge_weights = sum(edge.weight for edge in edges)
    max_edge_weight_per_stage = (
        sum_edge_weights * allowed_edge_imbalance / float(K)
    )
    for j in range(K):
        A = zeros(num_vars)
        for i in range(M):
            A[edge_var(i, j)] = edges[i].weight
        constraints.append(LinearConstraint(A=A, ub=max_edge_weight_per_stage))

    # Define the optimization objective:
    #   - the auxiliary variable equals to the maximum total edge weight in a stage;
    edge_weight_per_stage = sum_edge_weights / float(K)
    for j in range(K):
        A = zeros(num_vars)
        A[edge_aux_var] = -edge_weight_per_stage
        for i in range(M):
            A[node_var(edges[i].source, j)] += edges[i].weight
            A[node_var(edges[i].target, j)] += edges[i].weight
            A[edge_var(i, j)] = -edges[i].weight
        constraints.append(LinearConstraint(A=A, ub=0))

    #   - minimize the sum of inter-weight edges;
    c = zeros(num_vars)
    for i in range(M):
        for j in range(K):
            c[edge_var(i, j)] = -edges[i].weight
    c[edge_aux_var] = edge_weight_per_stage

    # Solve the MILP problem using scipy
    num_int_vars = num_node_vars + num_edge_vars
    integrality = torch.concatenate([ones(num_int_vars), zeros(num_aux_vars)])
    lb = torch.concatenate([zeros(num_int_vars), ones(num_aux_vars)])
    ub = torch.concatenate(
        [ones(num_int_vars), torch.full((num_aux_vars,), 2.0)]
    )
    bounds = Bounds(lb, ub)
    # Run the solver for at most that many seconds
    options = {"time_limit": SCIPY_TIME_LIMIT_SEC}
    start_time = time.time()
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options=options,
    )
    end_time = time.time()
    if result.status == 1:
        raise ValueError(
            "Iteration or time limit reached while solving the formulated MILP optimization."
            "Most likely this is due to inaccurately estimated ops memory footprints and "
            "communication costs. Try using an alternative pipeline splitting algorithm. "
        )
    if not result.success:
        raise ValueError(
            "The formulated MILP optimization is infeasible, likely due to inaccurately "
            "estimated ops memory footprints and communication costs. Try using an "
            "alternative pipeline splitting algorithm."
        )

    # Assign the resulting stages to each node
    for i in range(N):
        for j in range(K):
            # using a small threshold to avoid precision issues
            if abs(result.x[node_var(i, j)] - 1.0) < 1e-5:
                assert nodes[i].stage is None
                nodes[i].stage = j
        assert nodes[i].stage is not None

    logger.info(
        "Completed graph splitting in {:.1f} sec".format(end_time - start_time)
    )


def _print_splitting_stats(nodes, edges, num_stages):
    """Compute and print various stats related to graph splitting"""
    print(
        "Graph with |V| = {}, |E| = {}, |K| = {}".format(
            len(nodes), len(edges), num_stages
        )
    )
    sum_node_weights = sum(node.weight for node in nodes)
    node_bound_per_stage = sum_node_weights / float(num_stages)
    print(
        "Max allowed node weight per stage: {:,} ({})".format(
            int(sum_node_weights * MAX_MEMORY_IMBALANCE / float(num_stages)),
            MAX_MEMORY_IMBALANCE,
        )
    )
    sum_edge_weights = sum(edge.weight for edge in edges)
    edge_bound_per_stage = sum_edge_weights / float(num_stages)
    print(
        "Max allowed edge weight per stage: {:,} ({})".format(
            int(
                sum_edge_weights
                * MAX_COMMUNICATION_IMBALANCE
                / float(num_stages)
            ),
            MAX_COMMUNICATION_IMBALANCE,
        )
    )

    # Extract nodes/edges/weight per stage
    num_nodes: Dict[int, int] = defaultdict(int)
    num_edges: Dict[int, int] = defaultdict(int)
    node_weight: Dict[int, int] = defaultdict(int)
    edge_weight: Dict[int, int] = defaultdict(int)
    adj_weight: Dict[int, int] = defaultdict(int)

    for node in nodes:
        num_nodes[node.stage] += 1
        node_weight[node.stage] += node.weight

    cross_weight = 0
    cross_edges = 0
    cross_stage_weight = [
        [0 for _ in range(num_stages)] for _ in range(num_stages)
    ]
    for edge in edges:
        src_stage = nodes[edge.source].stage
        dst_stage = nodes[edge.target].stage
        if src_stage == dst_stage:
            num_edges[src_stage] += 1
            edge_weight[src_stage] += edge.weight
            adj_weight[src_stage] += edge.weight
        else:
            cross_weight += edge.weight
            cross_edges += 1
            cross_stage_weight[src_stage][dst_stage] += edge.weight
            adj_weight[src_stage] += edge.weight
            adj_weight[dst_stage] += edge.weight

    # Print the stats
    for stage in range(num_stages):
        print("  Stage {}:".format(stage), end="")
        print(
            " #nodes = {:3d} ({:.1f}%); node_weight = {:,} ({:5.1f}%);".format(
                num_nodes[stage],
                100.0 * num_nodes[stage] / len(nodes),
                node_weight[stage],
                100.0 * node_weight[stage] / node_bound_per_stage,
            ),
            end="",
        )
        print(
            " #edges = {:3d} ({:.1f}%);".format(
                num_edges[stage],
                100.0 * num_edges[stage] / len(edges),
            ),
            end="",
        )
        print(
            " edge_weight = {:,} ({:5.1f}%); adj_weight = {:,} ({:5.1f}%)".format(
                edge_weight[stage],
                100.0 * edge_weight[stage] / edge_bound_per_stage,
                adj_weight[stage],
                100.0 * adj_weight[stage] / edge_bound_per_stage,
            )
        )

    print(
        "cross-stage edges = {:}; cross-stage weight = {:,}".format(
            cross_edges, cross_weight
        )
    )
    for i in range(num_stages):
        for j in range(num_stages):
            if cross_stage_weight[i][j] == 0:
                continue
            print("  [{} -> {}] = {:,}".format(i, j, cross_stage_weight[i][j]))
