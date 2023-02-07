import copy
import itertools
import logging
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import fx

from torch.fx.graph import PythonCode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .graph_utils import get_node_tensor_metadata
from .profiler import GraphProfiler, GraphType, ProfilerEngine


class DistributedGraph:
    def __init__(
        self,
        orig_module: Optional[nn.Module] = None,
    ) -> None:
        self.orig_module: Optional[nn.Module] = orig_module
        self.graph_module: Optional[fx.GraphModule] = None
        self.fwd_graph_modules: List[fx.GraphModule] = []
        self.fwd_profilers: List[GraphProfiler] = []
        self.bwd_graph_modules: List[fx.GraphModule] = []
        self.bwd_profilers: List[GraphProfiler] = []
        # The mapping information is required for graph optimization.
        # This can only be derived by users. There is no way (at this moment)
        # to infer this information based only on the fx graphs.
        # TODO: Ideally, it will be better if fx/AOTAutograd can provide a way
        # to access original param, instead of us keeping the following maps.
        self.primal_name_to_node: List[Dict[str, fx.Node]] = []
        self.primal_to_param: List[Dict[fx.Node, nn.Parameter]] = []
        self.grad_to_primal: List[Dict[fx.Node, fx.Node]] = []

        # Following attributes are the general graph information. Each attribute
        # is a dictionary that key is the graph module and value is some grpah
        # information for that graph module.
        self.name_to_node: Dict[fx.GraphModule, Dict[str, fx.Node]] = {}

        # Indicate `update()` must be called before applying any optimization.
        self._dirty = True

    def update(self) -> "DistributedGraph":
        """
        Utility to put graph module into a node map for easier adjustments.
        """
        if not self._dirty:
            return self

        for gm in itertools.chain(
            self.fwd_graph_modules, self.bwd_graph_modules
        ):
            self.name_to_node[gm] = {node.name: node for node in gm.graph.nodes}

        self._dirty = False
        self.validate()
        return self

    def profile(self, *args: Any, **kwargs: Any) -> "DistributedGraph":
        """
        Profile the given distributed graph. The arguments are the inputs
        of the module as a real run is required to do profiling.
        """
        # TODO(chienchin): fix how to get the correct forward_loss
        def forward_loss(
            module: nn.Module, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            return module(*args, **kwargs).sum()

        assert self.orig_module is not None
        engine = ProfilerEngine(
            self.orig_module,
            forward_loss,
            dist_fwd_gm=self.fwd_graph_modules[0],
            dist_bwd_gm=self.bwd_graph_modules[0],
        )
        engine.run(*args, **kwargs)
        engine.summarize(to_print=True)
        self.fwd_profilers.append(engine.profilers[0][GraphType.FORWARD])
        self.bwd_profilers.append(engine.profilers[0][GraphType.BACKWARD])

        return self

    def validate(self) -> None:
        assert (
            not self._dirty
        ), "The graph is modified but ``update()`` is not called to update the information."
        assert (
            len(self.fwd_graph_modules) == 1
        ), "DistributedGraph has not support multiple subgraphs yet."
        assert len(self.fwd_graph_modules) == len(self.bwd_graph_modules)

        assert (
            len(self.primal_name_to_node)
            == len(self.primal_to_param)
            == len(self.grad_to_primal)
        )
        assert len(self.primal_name_to_node) <= len(self.fwd_graph_modules)
        assert len(self.fwd_profilers) == len(self.bwd_profilers)
        assert len(self.fwd_profilers) <= len(self.fwd_graph_modules)

        for primal_to_param, grad_to_primal in itertools.zip_longest(
            self.primal_to_param, self.grad_to_primal
        ):
            assert primal_to_param is not None and grad_to_primal is not None
            for param, grad in zip(
                primal_to_param.values(), grad_to_primal.keys()
            ):
                grad_tensor_meta = get_node_tensor_metadata(grad)
                assert param is not None, primal_to_param
                assert grad_tensor_meta is not None
                assert param.shape == grad_tensor_meta.shape
        assert len(self.primal_to_param) == len(self.primal_name_to_node)


class DistributedFxGraph(fx.Graph):
    def __init__(
        self,
        orig_graph: fx.Graph,
        setup_graph: fx.Graph,
        cleanup_graph: fx.Graph,
        owning_module: Optional[fx.GraphModule] = None,
        tracer_cls: Optional[Type["fx.Tracer"]] = None,
        tracer_extras: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(owning_module, tracer_cls, tracer_extras)

        output_vals = self.graph_copy(orig_graph, {}, return_output_node=True)
        self._codegen = copy.deepcopy(orig_graph._codegen)
        assert isinstance(output_vals, tuple)
        output_val, old_output_val = output_vals
        self.output(output_val, type_expr=getattr(old_output_val, "type", None))

        self.setup_graph = setup_graph
        self.cleanup_graph = cleanup_graph
        self._all_graphs = (super(), self.setup_graph, self.cleanup_graph)

        self._setup_mapping: Dict[fx.Node, fx.Node] = {}
        self._cleanup_mapping: Dict[fx.Node, fx.Node] = {}

        for node, setup_node, cleanup_node in zip(
            self.nodes, self.setup_graph.nodes, self.cleanup_graph.nodes
        ):
            self._setup_mapping[node] = setup_node
            self._cleanup_mapping[node] = cleanup_node

        self.num_extra_output = 0

    def _lookup_node(self, node: fx.Node, graph: fx.Graph) -> Optional[fx.Node]:
        if graph == self.setup_graph:
            return self._setup_mapping.get(node, None)
        elif graph == self.cleanup_graph:
            return self._cleanup_mapping.get(node, None)
        return node

    def _insert_context(self, func: str, node: fx.Node):
        with ExitStack() as stack:
            for graph in self._all_graphs:
                if node:
                    actual_node = self._lookup_node(node, graph)
                    assert (
                        actual_node is not None
                    ), "Cannot handle None case now."
                else:
                    actual_node = node
                stack.enter_context(getattr(graph, func)(actual_node))
            yield

    @contextmanager
    def inserting_after(self, node):
        return self._insert_context("inserting_after", node)

    @contextmanager
    def inserting_before(self, node):
        return self._insert_context("inserting_before", node)

    @staticmethod
    def _is_sese_graph(nodes: List[fx.Node], graph: fx.Graph) -> bool:
        """
        Check if the given subgraph is a single-entry-single-exit (SESE)
        graph of the original fx graph.
        """
        all_nodes: Set[fx.Node] = set(nodes)
        for i, node in enumerate(nodes):
            pytree_args, _ = tree_flatten(node.args)
            pytree_kwargs, _ = tree_flatten(node.kwargs)
            for arg in itertools.chain(pytree_args, pytree_kwargs):
                if node not in all_nodes and i > 0:
                    return False
            if i == len(nodes) - 1:
                # TODO: the only users should be the output. Otherwise, we don't
                # know how to move this subgraph. We currently do not stricly
                # force this attribute because some test code has orphan nodes.
                if len(node.users) > 1:
                    return False
            else:
                for user in node.users:
                    if user not in all_nodes:
                        return False
        return True

    def _convert_sese_input_to_output(
        self, nodes: List[fx.Node], graph: fx.Graph, erase_node: bool
    ) -> None:
        for output in reversed(graph.nodes):
            if output.target == "output":
                break
        first_node = self._lookup_node(nodes[0], graph)
        # TODO: We currently assume there is only one input to simplify the
        # coding. We may have to deal with a more general case.
        sese_arguments = first_node.args[0]
        # TODO: Do we need to remove the output of the SESE subgraph?
        # new_output =  tuple(
        #   arg for arg in output.args if arg != nodes[-1]
        # ) + (arguments,)
        new_output = output.args + (sese_arguments,)
        if erase_node:
            for node in nodes:
                graph_node = self._lookup_node(node, graph)
                graph.erase_node(graph_node)
        graph.erase_node(output)
        graph.output(new_output)

    def move_to_next_iter_before(
        self, nodes: List[fx.Node], target_node: fx.Node
    ):
        if not self._is_sese_graph(nodes, self):
            raise ValueError(
                "The nodes for move_to_next_iter_before must form a SESE "
                "subgraph. The output of this subgraph should be the output "
                " of the whole graph."
            )

        # For the setup graph, no additional input is needed but additional
        # outputs will be created. The additional output represents the input of
        # the action to be moved to the next iteration -- main graph.
        self._convert_sese_input_to_output(
            nodes=nodes, graph=self.setup_graph, erase_node=True
        )

        # For the main graph, additional input will be created to represent
        # the output from the last iteration -- main graph or setup graph.
        # Additional output will also be generated to represent the input for
        # the next iteration -- the main graph or the cleanup graph.
        self._convert_sese_input_to_output(
            nodes=nodes, graph=self, erase_node=False
        )
        new_input_node = self.placeholder(nodes[0].name + "_input")
        nodes[0].args = (new_input_node,)
        for node in nodes:
            target_node.prepend(node)
        nodes[0].prepend(new_input_node)
        nodes[-1].users[target_node] = None

        # For the cleanup graph, additional input is required to get the output
        # from the last iteration -- main graph. Additional nodes are also
        # needed to perform the action moved from the last itertion.
        first_cleanup_node = self._lookup_node(node, self.cleanup_graph)
        arguments = first_cleanup_node.args[0]

        new_input_node = self.cleanup_graph.placeholder(
            nodes[0].name + "_input"
        )
        target_cleanup_node = self._lookup_node(target_node, self.cleanup_graph)
        node_mapping: Dict[fx.Node, fx.Node] = {}
        with self.cleanup_graph.inserting_before(target_cleanup_node):
            last_new_cleanup_node: Optional[fx.Node] = None
            for i, node in enumerate(nodes):
                cleanup_node = self._lookup_node(node, self.cleanup_graph)
                # TODO: generalize the node copy process. We only support
                # call_function now and trivial args, kwargs for the first node.
                if i == 0:
                    args = (new_input_node,)
                    kwargs = {}
                else:
                    args = tree_map(
                        lambda arg: node_mapping[arg]
                        if isinstance(arg, fx.Node)
                        else arg,
                        cleanup_node.args,
                    )
                    kwargs = tree_map(
                        lambda arg: node_mapping[arg]
                        if isinstance(arg, fx.Node)
                        else arg,
                        cleanup_node.kwargs,
                    )
                new_cleanup_node = self.cleanup_graph.call_function(
                    cleanup_node.target,
                    args,
                    kwargs,
                    cleanup_node.type,
                )
                if i == 0:
                    new_cleanup_node.prepend(new_input_node)
                node_mapping[cleanup_node] = new_cleanup_node
                last_new_cleanup_node = new_cleanup_node
            assert last_new_cleanup_node is not None
            # TODO: Figure out how to properly avoid dead code elimination that
            # clean up the newly added node properly. Right now, we manually
            # update the users of the last node of the new SESE graph even
            # though the target node does not use that node.
            last_new_cleanup_node.users[target_cleanup_node] = None

        self.num_extra_output += 1

    def call_function(
        self,
        the_function: Callable[..., Any],
        args: Optional[Tuple["Argument", ...]] = None,
        kwargs: Optional[Dict[str, "Argument"]] = None,
        type_expr: Optional[Any] = None,
    ) -> fx.Node:
        setup_args = tree_map(
            lambda arg: self._lookup_node(arg, self.setup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            args,
        )
        setup_kwargs = tree_map(
            lambda arg: self._lookup_node(arg, self.setup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            kwargs,
        )
        cleanup_args = tree_map(
            lambda arg: self._lookup_node(arg, self.cleanup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            args,
        )
        cleanup_kwargs = tree_map(
            lambda arg: self._lookup_node(arg, self.cleanup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            kwargs,
        )

        setup_node = self.setup_graph.call_function(
            the_function, setup_args, setup_kwargs, type_expr
        )
        main_node = super().call_function(the_function, args, kwargs, type_expr)
        cleanup_node = self.cleanup_graph.call_function(
            the_function, cleanup_args, cleanup_kwargs, type_expr
        )
        self._setup_mapping[main_node] = setup_node
        self._cleanup_mapping[main_node] = cleanup_node
        return main_node

    def lint(self) -> None:
        self.setup_graph.lint()
        super().lint()
        self.cleanup_graph.lint()


class DistributedGraphModule(nn.Module):
    def __init__(self, main_gm: fx.GraphModule) -> None:
        super().__init__()

        def _copy_gm(src: fx.GraphModule, graph: fx.Graph) -> fx.GraphModule:
            gm = fx.GraphModule(src, graph)
            gm.meta = getattr(graph, "meta", {})
            return gm

        self.setup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.cleanup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.main_gm = _copy_gm(
            main_gm,
            DistributedFxGraph(
                main_gm.graph, self.setup_gm.graph, self.cleanup_gm.graph
            ),
        )

        self._iter = 0
        self._max_iters = 0
        self._previous_output: Tuple[Any] = tuple()

    def setup(self, max_iters: int = 0) -> None:
        """
        This method is used to tell DistributedGraphModule (DGM) the iterations to
        train so that DGM knows which iteration is the last one and can do proper
        cleanup.
        """
        if max_iters <= 0:
            raise ValueError(f"Incorrect max_iters is set, {max_iters}")
        self._iter = 0
        self._max_iters = max_iters

    def _run(self, gm: fx.GraphModule, *args, **kwargs) -> Any:
        if self.main_gm.graph.num_extra_output > 0:
            # TODO: a general way to support different types of input and output.
            assert not kwargs, "Has not supported kwargs now."
            new_args = args + (self._previous_output)
            output = gm(*new_args, **kwargs)
            if self._iter < self._max_iters:
                assert isinstance(
                    output, tuple
                ), f"Only support tuple output now. {type(output)}"
                num_actual_output = (
                    len(output) - self.main_gm.graph.num_extra_output
                )
                assert num_actual_output > 0
                self._previous_output = output[num_actual_output:]
                output = output[:num_actual_output]
                if len(output) == 1:
                    output = output[0]
            return output
        else:
            return gm(*args, **kwargs)

    def _print_out_graph(self) -> None:
        logging.warning(f"Printing the three fx gm:")
        logging.warning(f"1. Setup gm:")
        logging.warning(f"{self.setup_gm.print_readable(False)}")
        logging.warning(f"2. Main gm:")
        logging.warning(f"{self.main_gm.print_readable(False)}")
        logging.warning(f"3. Cleanup gm:")
        logging.warning(f"{self.cleanup_gm.print_readable(False)}")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._iter += 1
        if self._iter == 1:
            self._print_out_graph()
            logging.warning("Using the setup graph")
            gm = self.setup_gm
        elif self._iter == self._max_iters:
            logging.warning("Using the cleanup graph")
            gm = self.cleanup_gm
        else:
            logging.warning("Using the main graph")
            gm = self.main_gm

        return self._run(gm, *args, **kwargs)

    @property
    def graph(self) -> DistributedFxGraph:
        return self.main_gm.graph

    def recompile(self) -> PythonCode:
        self.setup_gm.recompile()
        self.cleanup_gm.recompile()
        return self.main_gm.recompile()


# Following is a test code only
fake_all_reduce_counter = 0
fake_wait_counter = 0


def fake_all_reduce(gradients: List[torch.Tensor]) -> torch.Tensor:
    global fake_all_reduce_counter
    fake_all_reduce_counter += 1
    logging.warning(f"fake_all_reduce {fake_all_reduce_counter}")
    return torch.concat(gradients)


def fake_wait(wait_tensor: torch.Tensor) -> torch.Tensor:
    global fake_wait_counter
    fake_wait_counter += 1
    logging.warning(f"fake_wait_tensor {fake_wait_counter}")
    return torch.clone(wait_tensor)


def fake_comm_fusion(gm: DistributedGraphModule):
    # Add a fake node to represent allreduce

    for node in gm.graph.nodes:
        if node.name == "addmm_5":
            break

    with gm.graph.inserting_after(node):
        all_reduce_node = gm.graph.call_function(fake_all_reduce, ([node],))
    with gm.graph.inserting_after(all_reduce_node):
        wait_node = gm.graph.call_function(fake_wait, (all_reduce_node,))

    for target_node in gm.graph.nodes:
        if target_node.name == "addmm_4":
            break

    gm.graph.move_to_next_iter_before([wait_node], target_node)

    gm.graph.lint()
    gm.recompile()
