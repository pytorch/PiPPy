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
    memory_weight: int
    comm_weight: int
    stage: Optional[int]
    gm_nodes: List[fx.Node]

    def __hash__(self):
        return hash(self.name)


@dataclass
class Edge:
    source: int
    target: int
    comm_weight: int


MAX_MEMORY_IMBALANCE = 2.5
MAX_COMMUNICATION_IMBALANCE = 1.1
SCIPY_TIME_LIMIT_SEC = 30
PRESOLVE = True


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

    # Pre-process the input graph by merging pairs of nodes that need to be in
    # the same stage. This reduces the size of the instance for the main solver
    if PRESOLVE:
        nodes, edges = _split_presolve(nodes, edges)

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

    # Prepare the result
    result = {}
    for node in nodes:
        if node.stage is None:
            continue
        for gm_node in node.gm_nodes:
            result[gm_node] = node.stage
    return result


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
        if "val" in node.meta:
            tensors = node.meta["val"]
            if isinstance(tensors, torch.Tensor):
                tensors = [tensors]
            activation_size[node] = sum(t.numel() for t in tensors)
        else:
            activation_size[node] = 0
        node_index[node.name] = len(nodes)
        nodes.append(Node(node.name, weight, 0, None, [node]))

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
        sum(node.memory_weight for node in nodes) > 0
    ), "node weights cannot be empty"
    assert (
        sum(edge.comm_weight for edge in edges) > 0
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
        "Splitting a graph with {} nodes and {} edges into {} stages".format(
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
    #   - edges go from a lower-index stage to an upper-index stage;
    multiplier = [2 ** (K - j - 1) for j in range(K)]
    for i in range(M):
        edge = edges[i]
        A = zeros(num_vars)
        for j in range(K):
            A[node_var(edge.target, j)] = multiplier[j]
            A[node_var(edge.source, j)] = -multiplier[j]
        constraints.append(LinearConstraint(A=A, ub=0))

    # Constraint 4:
    #   - nodes in every stage have (approximately) the same total weight;
    sum_node_weights = sum(node.memory_weight for node in nodes)
    max_node_weight_per_stage = (
        sum_node_weights * allowed_node_imbalance / float(K)
    )
    for j in range(K):
        A = zeros(num_vars)
        for i in range(N):
            A[node_var(i, j)] = nodes[i].memory_weight
        constraints.append(LinearConstraint(A=A, ub=max_node_weight_per_stage))

    # Constraint 5:
    #   - edges in every stage have (approximately) the same total weight;
    sum_edge_weights = sum(edge.comm_weight for edge in edges) + sum(
        node.comm_weight for node in nodes
    )
    max_edge_weight_per_stage = (
        sum_edge_weights * allowed_edge_imbalance / float(K)
    )
    for j in range(K):
        A = zeros(num_vars)
        for i in range(M):
            A[edge_var(i, j)] = edges[i].comm_weight
        for i in range(N):
            A[node_var(i, j)] = nodes[i].comm_weight
        constraints.append(LinearConstraint(A=A, ub=max_edge_weight_per_stage))

    # Define the optimization objective:
    #   - the auxiliary variable equals to the maximum total edge-weight in a stage;
    edge_weight_per_stage = sum_edge_weights / float(K)
    for j in range(K):
        A = zeros(num_vars)
        A[edge_aux_var] = -edge_weight_per_stage
        for i in range(M):
            edge = edges[i]
            A[node_var(edge.source, j)] += edge.comm_weight
            A[node_var(edge.target, j)] += edge.comm_weight
            A[edge_var(i, j)] = -edge.comm_weight
        for i in range(N):
            A[node_var(i, j)] += nodes[i].comm_weight
        constraints.append(LinearConstraint(A=A, ub=0))

    #   - minimize the sum of inter-weight edges;
    c = zeros(num_vars)
    for j in range(K):
        for i in range(M):
            c[edge_var(i, j)] = -edges[i].comm_weight - 1
        for i in range(N):
            c[node_var(i, j)] = -nodes[i].comm_weight
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


"""
Pre-solve the splitting problem by merging nodes that needs to be in the same
stage. The algorithm works by
"""


def _split_presolve(nodes: List[Node], edges: List[Edge]):
    # Count the in- and out- degree of each node
    in_degree: Dict[Node, int] = defaultdict(int)
    out_degree: Dict[Node, int] = defaultdict(int)
    for edge in edges:
        out_degree[nodes[edge.source]] += 1
        in_degree[nodes[edge.target]] += 1

    # Initialize singleton clusters
    clusters: List[List[Node]] = []
    node2cluster: Dict[Node, int] = defaultdict(int)
    for node in nodes:
        node2cluster[node] = len(clusters)
        clusters.append([node])

    def should_merge_edge(src, dst):
        """Decide whether the edge src->dst should be merged at pre-solving"""
        # already merged
        if node2cluster[src] == node2cluster[dst]:
            return False
        # always merge sources having a unique successor
        if in_degree[src] == 0 and out_degree[src] == 1:
            return True
        # always merge sinks having a unique predecessor
        if in_degree[dst] == 1 and out_degree[dst] == 0:
            return True
        # merge chains of degree-1 nodes
        if in_degree[src] == 1 and out_degree[src] == 1 and in_degree[dst] == 1:
            return True
        return False

    # Merge edges in the decreasing order of their weight
    sorted_edges = sorted(edges, key=lambda e: e.comm_weight, reverse=True)
    for edge in sorted_edges:
        src = nodes[edge.source]
        dst = nodes[edge.target]
        if not should_merge_edge(src, dst):
            continue
        cluster_src = clusters[node2cluster[src]]
        cluster_dst = clusters[node2cluster[dst]]
        cluster_src.extend(cluster_dst)
        for node_dst in cluster_dst:
            node2cluster[node_dst] = node2cluster[src]
        cluster_dst.clear()

    # Collect the resulting nodes
    merged_nodes: List[Node] = []
    node_index = {}
    for chain_idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        name = cluster[0].name
        gm_nodes = []
        for node in cluster:
            node_index[node.name] = len(merged_nodes)
            gm_nodes.extend(node.gm_nodes)
        mem_weight = sum(node.memory_weight for node in cluster)
        comm_weight = sum(
            edge.comm_weight
            for edge in edges
            if nodes[edge.source] in cluster and nodes[edge.target] in cluster
        )
        merged_nodes.append(Node(name, mem_weight, comm_weight, None, gm_nodes))

    # Collect the resulting edges
    merged_edges: List[Edge] = []
    for edge in edges:
        src = nodes[edge.source]
        dst = nodes[edge.target]
        if node2cluster[src] == node2cluster[dst]:
            continue
        source_idx = node_index[src.name]
        target_idx = node_index[dst.name]
        merged_edges.append(Edge(source_idx, target_idx, edge.comm_weight))

    logger.info(
        "merged {} nodes and {} edges; max cluster has size {}".format(
            len(nodes) - len(merged_nodes),
            len(edges) - len(merged_edges),
            max(len(c) for c in clusters),
        )
    )
    return merged_nodes, merged_edges


def _print_splitting_stats(nodes, edges, num_stages):
    """Compute and print various stats related to graph splitting"""
    print(
        "Graph with |V| = {}, |E| = {}, |K| = {}".format(
            len(nodes), len(edges), num_stages
        )
    )
    sum_node_weights = sum(node.memory_weight for node in nodes)
    node_bound_per_stage = sum_node_weights / float(num_stages)
    print(
        "Max allowed node weight per stage: {:,} ({})".format(
            int(sum_node_weights * MAX_MEMORY_IMBALANCE / float(num_stages)),
            MAX_MEMORY_IMBALANCE,
        )
    )
    sum_edge_weights = sum(edge.comm_weight for edge in edges) + sum(
        node.comm_weight for node in nodes
    )
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
    mem_weight: Dict[int, int] = defaultdict(int)
    com_weight: Dict[int, int] = defaultdict(int)
    adj_weight: Dict[int, int] = defaultdict(int)

    for node in nodes:
        num_nodes[node.stage] += len(node.gm_nodes)
        mem_weight[node.stage] += node.memory_weight
        com_weight[node.stage] += node.comm_weight
        adj_weight[node.stage] += node.comm_weight

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
            com_weight[src_stage] += edge.comm_weight
            adj_weight[src_stage] += edge.comm_weight
        else:
            cross_weight += edge.comm_weight
            cross_edges += 1
            cross_stage_weight[src_stage][dst_stage] += edge.comm_weight
            adj_weight[src_stage] += edge.comm_weight
            adj_weight[dst_stage] += edge.comm_weight

    # Print the stats
    total_nodes = sum(len(node.gm_nodes) for node in nodes)
    for stage in range(num_stages):
        print("  Stage {}:".format(stage), end="")
        print(
            " #nodes = {:3d} ({:.1f}%); mem_weight = {:,} ({:5.1f}%);".format(
                num_nodes[stage],
                100.0 * num_nodes[stage] / total_nodes,
                mem_weight[stage],
                100.0 * mem_weight[stage] / node_bound_per_stage,
            ),
            end="",
        )
        print(
            " #edges = {:3d} ({:4.1f}%);".format(
                num_edges[stage],
                100.0 * num_edges[stage] / len(edges),
            ),
            end="",
        )
        print(
            " com_weight = {:,} ({:5.1f}%); adj_weight = {:,} ({:5.1f}%)".format(
                com_weight[stage],
                100.0 * com_weight[stage] / edge_bound_per_stage,
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
