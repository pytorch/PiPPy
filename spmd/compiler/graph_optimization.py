from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterable, Set

from .bucketing_strategies import BucketingStrategy
from .distributed_graph import DistributedGraph
from .scheduling_policies import SchedulingPolicy


# It is more nature to set a run after list for a function decorator, but
# it is easier to have check with run_before_set.
_run_before_sets: DefaultDict[str, Set[str]] = defaultdict(set)


def graph_optimization_pass(run_after: Iterable[str] = tuple()) -> Any:
    """
    The contract of graph optimization pass. All the passes should be wrapped with
    this decorator. The first argument of a pass is the target DistGraphOptimization,
    and the return value must be the instanceg. All the modifications are in-place.
    Users can then chain the passes to optimize the whole graph.
    """

    def pass_wrapper(func: Any) -> Any:
        for name in run_after:
            _run_before_sets[name].add(func.__name__)

        def pass_func(
            self: "DistGraphOptimization", *args: Any, **kwargs: Any
        ) -> "DistGraphOptimization":
            assert not (
                self._optimized or func.__name__ in self._optimized_func
            ), f"Cannot apply {func.__name__} twice."
            self._optimizing = True
            invalid_passes = _run_before_sets[func.__name__].intersection(
                self._optimized_func
            )
            assert (
                not invalid_passes
            ), f"{invalid_passes} must run after {func.__name__}"
            self.graph.validate()

            ret = func(self.graph, *args, **kwargs)

            assert self.graph == ret
            self.graph.validate()
            self._optimized_func.add(func.__name__)
            return self

        return pass_func

    return pass_wrapper


class DistGraphOptimization:
    """
    The class to hold the optimization passes to allow users to chain the passes.

    Example:
        graph = DistGraphOptimization(
            graph=graph
        ).fuse_communication(
            bucketing_strategy, scheduling_policy,
        ).another_optimization_pass(
            *args, **kwargs
        ).graph
    """

    def __init__(self, graph: DistributedGraph) -> None:
        self._optimizing = False
        self._optimized = False
        self._optimized_func: Set[str] = set()
        self._graph = graph

    @property
    def graph(self) -> DistributedGraph:
        if self._optimizing:
            self._optimized = True
            self._optimizing = False
        return self._graph

    @property
    def optimized(self) -> bool:
        return self._optimized

    @graph_optimization_pass()
    def fuse_communication(
        self,
        bucketing_strategy: BucketingStrategy,
        scheduling_policy: SchedulingPolicy,
    ) -> "DistGraphOptimization":
        return self
