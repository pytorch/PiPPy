class _ModuleFrame:
    def __init__(
        self,
        flat_graph,
        nodes,
        seen_nodes,
        seen_modules,
        parent,
        module_stack,
        module_id,
        module_call_graph: Dict[str, ModuleCallSignature],
        module: Optional[torch.nn.Module] = None,
    ):
        self.flat_graph = flat_graph
        self.nodes = nodes
        self.seen_nodes = seen_nodes
        self.seen_modules = seen_modules
        self.parent = parent
        self.module_stack = module_stack
        self.module_id = module_id

        self.module_call_graph = module_call_graph
        self.verbose = False

        self.fqn = self.module_stack[-1]
        if module is not None:
            self.module = module
        else:
            self.module = InterpreterModule(torch.fx.Graph())
        if self.module_id in self.seen_modules:
            self.cached_graph_module = self.seen_modules[self.module_id]
        else:
            self.cached_graph_module = None
            self.seen_modules[self.module_id] = self.module

        self.graph = self.module.graph

        # Mapping of nodes in the flat graph to nodes in this graph.
        self.node_map: Dict[torch.fx.Node, torch.fx.Node] = {}
        self.node_to_placeholder = {}

        self.parent_call_module: Optional[torch.fx.Node] = None
        if parent is not None:
            accessor = _compute_accessor(parent.fqn, self.fqn)
            _add_submodule(
                parent.module,
                accessor,
                self.module
                if self.cached_graph_module is None
                else self.cached_graph_module,
            )
            self.parent_call_module = parent.graph.call_module(accessor)

        signature = module_call_graph.get(self.fqn)
        if signature is not None and self.parent is not None:
            assert signature.in_spec.num_children == 2
            args_spec = signature.in_spec.children_specs[0]
            kwargs_spec = signature.in_spec.children_specs[1]
            assert args_spec.context is None
            assert kwargs_spec.context is not None

            with self.graph.inserting_after(None):
                arg_nodes = []
                for idx in range(args_spec.num_children):
                    arg_nodes.append(self.graph.placeholder(f"_positional_arg_{idx}"))
                kwarg_nodes = {}
                for name in kwargs_spec.context:
                    kwarg_nodes[name] = self.graph.placeholder(name)
                flat_args = _generate_flatten(
                    self.module,
                    (tuple(arg_nodes), kwarg_nodes),
                    signature.in_spec,
                )
                for idx, arg in enumerate(signature.inputs):
                    flat_arg_node = self.graph.create_node(
                        op="call_function",
                        target=operator.getitem,
                        args=(flat_args, idx),
                        name=arg.name
                        if not isinstance(arg, ConstantArgument)
                        else f"_constant_{idx}",
                    )
                    if isinstance(arg, ConstantArgument):
                        continue
                    flat_arg_node.meta = copy.copy(self.seen_nodes[arg.name].meta)
                    self.node_to_placeholder[self.seen_nodes[arg.name]] = flat_arg_node

            with self.parent.graph.inserting_before(self.parent_call_module):
                input_nodes: List[Optional[torch.fx.Node]] = []
                for input in signature.inputs:
                    if isinstance(input, ConstantArgument) and input.value is None:
                        input_nodes.append(None)
                    else:
                        assert isinstance(input, (TensorArgument, SymIntArgument))
                        input_nodes.append(
                            self.parent.remap_input(self.seen_nodes[input.name])
                        )

                inputs_node = _generate_unflatten(
                    self.parent.module,
                    input_nodes,
                    signature.in_spec,
                )

                args_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 0)
                )
                kwargs_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 1)
                )
                arg_nodes = [
                    self.parent.graph.call_function(operator.getitem, (args_node, i))
                    for i in range(args_spec.num_children)
                ]
                kwarg_nodes = {
                    k: self.parent.graph.call_function(
                        operator.getitem, (kwargs_node, k)
                    )
                    for k in kwargs_spec.context
                }
            assert self.parent_call_module is not None
            self.parent_call_module.args = tuple(arg_nodes)
            self.parent_call_module.kwargs = kwarg_nodes

    def add_placeholder(self, x):
        assert x.graph is self.flat_graph
        # x is not in subgraph, create a new placeholder for subgraph
        with self.graph.inserting_before(None):
            placeholder_node = self.graph.placeholder(x.name, type_expr=x.type)
        # copy all meta fields, even if some fields might be irrelvant for
        # the placeholder node
        placeholder_node.meta = copy.copy(x.meta)
        self.node_to_placeholder[x] = placeholder_node

    def remap_input(self, x):
        assert x.graph is self.flat_graph
        if x in self.node_map:
            return self.node_map[x]
        if x not in self.node_to_placeholder:
            self.add_placeholder(x)
            if self.parent_call_module is not None:
                # Important to *prepend* the output to match how we are
                # inserting placeholder nodes.
                self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
        return self.node_to_placeholder[x]

    def finalize_outputs(self):
        orig_outputs = []

        signature = self.module_call_graph.get(self.fqn)
        if signature is not None and self.parent is not None:
            for output in signature.outputs:
                if isinstance(output, (TensorArgument, SymIntArgument)):
                    orig_outputs.append(self.seen_nodes[output.name])
                else:
                    raise RuntimeError(
                        f"Unsupported data type for output node: {output}"
                    )

            tree_out_node = _generate_unflatten(
                self.module,
                tuple(
                    self.node_map[self.seen_nodes[output.name]]
                    for output in orig_outputs
                ),
                signature.out_spec,
            )
            parent_out: Optional[torch.fx.Node] = _generate_flatten(
                self.parent.module, self.parent_call_module, signature.out_spec
            )
            graph_outputs: Union[torch.fx.Node, List[torch.fx.Node]] = tree_out_node
        else:
            graph_outputs = []
            # Iterate through nodes we have copied into self.graph.
            for orig_node in self.node_map.keys():
                for user_node in orig_node.users:
                    if user_node.name not in self.seen_nodes:
                        # external user node, need to expose as an output
                        orig_outputs.append(orig_node)
                        graph_outputs.append(self.node_map[orig_node])
                        break

            parent_out = self.parent_call_module
            if len(graph_outputs) == 1:
                graph_outputs = graph_outputs[0]

        assert isinstance(graph_outputs, (list, torch.fx.Node))

        self.graph.output(graph_outputs)

        # Rewrite outputs in parent module
        if parent_out is None:
            return

        parent_out.meta["val"] = (
            graph_outputs.meta.get("val")
            if isinstance(graph_outputs, torch.fx.Node)
            else [o.meta.get("val") for o in graph_outputs]
        )

        if len(orig_outputs) == 1 and signature is None:
            self.parent.node_map[orig_outputs[0]] = parent_out
        else:
            for i, orig_output in enumerate(orig_outputs):
                # Use Proxy to record getitem access.
                proxy_out = torch.fx.Proxy(parent_out)[i].node  # type: ignore[index]
                proxy_out.meta["val"] = orig_output.meta.get("val")
                self.parent.node_map[orig_output] = proxy_out

        if self.cached_graph_module is not None:
            _verify_graph_equivalence(self.cached_graph_module, self.module)

    def copy_node(self, node):
        self.print("copying", node.format_node())
        self.node_map[node] = self.graph.node_copy(node, self.remap_input)
        self.seen_nodes[node.name] = node

    def run_outer(self):
        i = 0
        for node in self.flat_graph.nodes:
            self.print(i, node.meta.get("nn_module_stack"), node.format_node())
            i += 1

        # Copy all graph inputs
        node_idx: int = 0
        node = self.nodes[node_idx]
        while node.op == "placeholder":
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]

        self.run_from(node_idx)

        # Copy graph outputs
        for node in self.flat_graph.nodes:
            if node.op == "output":
                self.copy_node(node)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def run_from(self, node_idx):
        module_idx = 0
        # Walk through the graph, building up a new graph with the right submodules
        while node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            assert node.op != "placeholder"

            self.print()
            self.print("STEP", node_idx, node.format_node())
            self.print(self.module_stack)
            if node.op == "output":
                if len(self.module_stack) == 1:
                    # We want the output node of the original graph to be handled
                    # specially by the outermost stack frame (in run_outer). So
                    # skip finalization here.
                    return node_idx

                # We've reached the end of the graph. Wrap up all the existing stack frames.
                self.finalize_outputs()
                return node_idx

            node_module_stack = (
                [path for path, ty in node.meta["nn_module_stack"].values()]
                if "nn_module_stack" in node.meta
                else self.module_stack
            )
            if node_module_stack[: len(self.module_stack)] != self.module_stack:
                # This means that the current module is done executing and the
                # current node is the beginning of a new module.
                #
                # In this case, we should finalize this module and return without
                # incrementing the node counter.
                self.finalize_outputs()
                self.print("outlining", self.fqn)
                self.print(self.graph)
                return node_idx

            assert node_module_stack is not None

            if _is_prefix(self.module_stack, node_module_stack):
                # This means that the current node represents the execution of a new
                # module.
                next_module = node_module_stack[len(self.module_stack)]
                self.print("Creating new stack frame for", next_module)
                # Run a nested version of module outliner from the current node
                # counter. Once it is complete, continue from that point.
                node_idx = _ModuleFrame(
                    self.flat_graph,
                    self.nodes,
                    self.seen_nodes,
                    self.seen_modules,
                    self,
                    self.module_stack + [next_module],
                    list(node.meta["nn_module_stack"].keys())[len(self.module_stack)],
                    self.module_call_graph,
                ).run_from(node_idx)
                module_idx += 1
                continue

            # The only remaining possibility is that we are in the right stack
            # frame. Copy the node into this frame's graph and increment the node counter.
            assert node_module_stack == self.module_stack
            self.copy_node(node)
            node_idx += 1


def _outline_submodules(orig_graph: torch.fx.Graph, root_module: UnflattenedModule):
    seen_nodes: Dict[str, torch.fx.Node] = {}
    seen_modules: Dict[int, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        None,
        [""],
        "",
        {
            entry.fqn: entry.signature
            for entry in root_module.module_call_graph
            if entry.signature
        },
        module=root_module,
    ).run_outer()
