# [WIPPy] PiPPy: Pipeline Parallelism for PyTorch

This project is an attempt to build a state-of-the-art automated Pipeline Parallelism system for PyTorch subject to the design considerations brought up in these [RFCs](https://github.com/pytorch/rfcs/pull/32). Some of the main design considerations include:

* An eye toward making running PyTorch code under Pipeline Parallelism as seamless as possible; that is, the user should have to make as few changes to their code as possible. In particular, we wish to elide the requirement of existing systems to structure your code as an `nn.Sequential`.
* First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects). This is currently missing from the torchgpipe-based `torch.distributed.pipeline.sync.Pipe`
* Composability with other parallelism schemes such as data parallelism or tensor splitting model parallelism (overall, known as "3d parallelism")
* Support for pipeline scheduling paradigms, including static schedules like fill-drain (GPipe), 1f1b, interleaved 1f1b and dynamic schedules like lookahead or registers/back-pressure.
* Support non-trivial topologies, including skip connections and tied weights/layers.

# Design and Codebase Roadmap

## Program Capture and Intermediate Representation

`IR.py` defines the `Pipe` class, which is the main intermediate representation used in PiPPy. This intermediate representation consists of a restricted `fx.GraphModule`. In the top level `fx.Graph` representation, the IR is limited to only the following node types:
  * `placeholder` and `output` nodes to specify overall pipeline inputs and outputs, respectively
  * `call_module` nodes to represents calls into the pipeline stages
    * A single call to the `_loss` submodule can also be present to represent the loss computation for training
  * `call_function` with a target of `operator.getitem`, for unpacking tuple outputs from pipeline stages
  * `call_function` with a target of `IR.stage_backward` for the purpose of modeling the backward computation of each pipeline stage. This is described later in the section about IR for the backward pass.
  * `call_function` with a target of `torch.add`, emitted solely for accumulating gradients of values that have multiple uses in the backward pass.

The top-level `fx.Graph` gives us 1) a topological ordering of pipeline stages and 2) the data dependencies between these pipeline stages. Note that this is more general than existing pipeline APIs, as it supports arbitrary non-local (i.e. skip) connections between stages.

We can create IR from existing PyTorch modules using one of several front-ends, exposed as static methods on `Pipe`. `Pipe.from_sequential` takes as argument an instance of `torch.nn.Sequential` and returns a `Pipe` instance that represents the trivial feed-forward nature of that sequential. For example:

```
mods = [torch.nn.Linear(512, 512) for _ in range(5)]
mods += [mods[0]]
seq = torch.nn.Sequential(*mods)

seq_pipe = Pipe.from_sequential(seq)

print(seq_pipe.split_gm)
"""
GraphModule(
  (submod_0): Linear(in_features=512, out_features=512, bias=True)
  (submod_1): Linear(in_features=512, out_features=512, bias=True)
  (submod_2): Linear(in_features=512, out_features=512, bias=True)
  (submod_3): Linear(in_features=512, out_features=512, bias=True)
  (submod_4): Linear(in_features=512, out_features=512, bias=True)
  (submod_5): Linear(in_features=512, out_features=512, bias=True)
)



def forward(self, input):
    input_1 = input
    _0 = self.submod_0(input_1);  input_1 = None
    _1 = self.submod_1(_0);  _0 = None
    _2 = self.submod_2(_1);  _1 = None
    _3 = self.submod_3(_2);  _2 = None
    _4 = self.submod_4(_3);  _3 = None
    _5 = self.submod_5(_4);  _4 = None
    return _5
```

Similarly, we can use `Pipe.from_tracing` to use `torch.fx` tracing to convert an arbitrary `nn.Module` instance to this form. For example:

```
class ExampleCode(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mm_param = torch.nn.Parameter(torch.randn(512, 512))
    self.mm_param2 = torch.nn.Parameter(torch.randn(512, 512))
    self.lin = torch.nn.Linear(512, 512)

  def forward(self, x):
    x = torch.mm(x, self.mm_param)
    skip_connection = x
    x = torch.relu(x)
    pipe_split()
    x = torch.mm(x, self.mm_param)
    x = self.lin(x)
    pipe_split()
    x = torch.relu(x)
    x = x + skip_connection
    x = torch.mm(x, self.mm_param2)
    x = self.lin(x)
    return x

ec = ExampleCode()
ec(torch.randn(50, 512))

ec_pipe = Pipe.from_tracing(ec, MultiUseParameterConfig.TRANSMIT)
print(ec_pipe.split_gm)
"""
GraphModule(
  (submod_0): GraphModule()
  (submod_1): GraphModule(
    (lin): Linear(in_features=512, out_features=512, bias=True)
  )
  (submod_2): GraphModule(
    (lin): Linear(in_features=512, out_features=512, bias=True)
  )
)



def forward(self, x):
    submod_0 = self.submod_0(x);  x = None
    getitem_2 = submod_0[2]
    getitem = submod_0[0]
    getitem_1 = submod_0[1];  submod_0 = None
    submod_1 = self.submod_1(getitem, getitem_2);  getitem = getitem_2 = None
    submod_2 = self.submod_2(submod_1, getitem_1);  submod_1 = getitem_1 = None
    return submod_2
```

There are a few things to note about the above example:

1. We use `IR.pipe_split` to explicitly demarcate within the code where we want pipeline boundaries to be. `from_tracing` will collect all data dependencies across these calls to `pipe_split` and emit corresponding data dependencies in the pipeline graph.
    * Note that `IR.PipeSplitWrapper` and `IR.annotate_split_points` can be used to unintrusively specify split points at the beginning or end of execution of any Module in the module hierarchy
2. Note the `skip_connection` value in the original program. `from_tracing` will correctly detect the usage of this value in non-adjacent pipeline stages and emit a connection in the top-level graph to forward this dependency from stage 0 to 2.
3. Notice that `self.mm_param` is used both in pipeline stage 0 and pipeline stage 1. Since we have specified `MultiUseParameterConfig.TRANSMIT` as the `multi_use_param_spec` argument to `from_tracing`, the system will emit code that will keep `mm_param` resident on stage 0 and transmit that value for use within stage 1. `multi_use_param_spec` can also be specified as a dictionary mapping parameter qualified names to a `MultiUseParameterConfig` value (one of `TRANSMIT` or `REPLICATE`) or it can be left as `None` to specify the default behavior (`TRANSMIT`) for all shared parameters. We will discuss replication in the following section.


Multi-use parameters can also be replicated. That is, each pipeline stage that uses a replicated parameter will have its own copy of the parameter and the system will record information about this replication such that the runtime can insert the proper synchronization operations upon update of these parameters. For example, let us rerun the above example with `multi_use_param_spec=MultiUseParameterConfig.REPLICATE`:

```
ec_pipe_replicated = Pipe.from_tracing(ec, MultiUseParameterConfig.REPLICATE)
print(ec_pipe_replicated.replicated_params)
"""
[{'submod_0': '__mm_param', 'submod_1': '__mm_param'},
 {'submod_1': 'lin.weight', 'submod_2': 'lin.weight'},
 {'submod_1': 'lin.bias', 'submod_2': 'lin.bias'}]
"""
```

Note that the `Pipe` instance has an attribute `replicated_params`, which is a record of all of the parameters that are replicated across pipeline stages. This object is a list of dictionaries. Each dictionary represents a single value that has been replicated across stages. The keys of the dictionary are the qualified name of the pipeline stage submodules that hold copies of this parameter, and the values are the qualified name of the parameter itself within those pipeline stage modules. Note that not only do we see `mm_param` in the above example, but we also see parameter replication from the usage of the `self.lin` module in multiple pipeline stages. `self.lin` is a "leaf module" in `torch.fx` parlance, and since we cannot see into the implementation of a leaf module, we automatically replicate leaf module parameters (note that we could hypothetically emit code to fetch the parameters values from leaf modules and transmit them to use sites, but that will require further development work).

### Aside: Futher Considerations for Program Capture

* `torch.fx` tracing imposes limitations on the classes of programs that can be captured (as described in [Limitations of Symbolic Tracing](https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing)). Thus, this limits the applicability of the above-described system. However, we can think of several ways to address this:
    * Convert tracing into a "just-in-time" system, where program capture happens on each invocation, and is specialized to certain parameters like shapes flowing throughout the program. By using ephemeral traces that are transmitted to each worker on each invocation, we can address the limitations of e.g. dynamic control flow. However, we will need to figure out the semantics of parameter placement in this scenario, as moving those around on each invocation will likely be sub-optimal
    * We can formulate pipeline paralellism as a program (program counter, call stack, live heap content) that migrates through multiple devices. Thus, rather than defining semantics for program capture and analysis, we simply need to define runtime semantics for migrating a running coroutine between devices. This may be difficult to implement in existing languages (Python). This could be implemented in TorchScript, but then that would require the program to be admissible to TorchScript's program capture limitations. Maybe we should just make a new language.

## Intermediate Representation: Loss and backward() Computation

`Pipe.from_sequential` and `Pipe.from_tracing` also take a `loss_fn` argument to specify the loss computation in the training scenario. `loss_fn` can be an `nn.Module` instance or a free function. The module/function should take two positional arguments: the output of the feedforward computation and the `target` values. An example of using this API and the IR it produces can be seen here:

```
ec_pipe_with_loss = Pipe.from_tracing(ec, loss_fn=mse_loss)
print(ec_pipe_with_loss.split_gm)
"""
GraphModule(
  (submod_0): GraphModule()
  (submod_1): GraphModule(
    (lin): Linear(in_features=512, out_features=512, bias=True)
  )
  (submod_2): GraphModule(
    (lin): Linear(in_features=512, out_features=512, bias=True)
  )
  (_loss): MSELoss()
)



def forward(self, x, target):
    submod_0 = self.submod_0(x)
    getitem_2 = submod_0[2]
    getitem = submod_0[0]
    getitem_1 = submod_0[1]
    submod_1 = self.submod_1(getitem, getitem_2)
    submod_2 = self.submod_2(submod_1, getitem_1)
    _loss = self._loss(submod_2, target)
    stage_backward = __main___stage_backward(stage_output = _loss, output_grads = None, input_values = [submod_2, target]);  target = None
    getitem_3 = stage_backward[0]
    getitem_4 = stage_backward[1];  stage_backward = None
    stage_backward_1 = __main___stage_backward(stage_output = submod_2, output_grads = getitem_3, input_values = [submod_1, getitem_1]);  submod_2 = getitem_3 = getitem_1 = None
    getitem_5 = stage_backward_1[0]
    getitem_6 = stage_backward_1[1];  stage_backward_1 = None
    stage_backward_2 = __main___stage_backward(stage_output = submod_1, output_grads = getitem_5, input_values = [getitem, getitem_2]);  submod_1 = getitem_5 = getitem = getitem_2 = None
    getitem_7 = stage_backward_2[0]
    getitem_8 = stage_backward_2[1];  stage_backward_2 = None
    stage_backward_3 = __main___stage_backward(stage_output = submod_0, output_grads = [getitem_7, getitem_6, getitem_8], input_values = [x]);  submod_0 = getitem_7 = getitem_6 = getitem_8 = x = None
    return _loss
"""
```

Note the following:

1. When `loss_fn` is specified, an additional positional input (`target`) is added to the signature of the model. During training, the value representing the target label used in loss computation should be passed in as this argument.
2. The loss value is returned, allowing for e.g. logging of loss values during training.
3. A simple symbolic automatic differentiation process emits code for computing the gradients of the model training process in a pipelined way. This is described below.

## Symbolic Autodiff

When thinking about how to implement backwards() in a pipeline parallel scenario, there are two considerations:


* In PyTorch autograd, `requires_grad` dictates whether operations record values/functions in the autograd tape. However, this does not compel execution of the backward for those operations.
* `backward` is only run when a corresponding `backward()` call later in the program initiates gradient computation.


Given this, we should think about how these semantics could map onto pipeline parallel execution, especially given that we would like to arbitrarily schedule the forward/backward jobs in the computation (such as in 1f1b scheduling). There are basically two options here:


1. Emulate the “autograd tracing + just-in-time gradient computation” of Eager mode PyTorch. This may be a bit difficult to do in conjunction with schedules, as the dynamic nature of this type of autograd makes it more difficult to reason about if/when/what is run in autograd or to reference specific forward/backward executions in a schedule
2. Model `backward` ahead-of-time by emitting stages into the IR. This allows us to schedule all stages uniformly (we don’t need to essentially replicate the autograd machinery in the runtime) and allows us to know what backward stages will be run ahead of time and reference them in the scheduling system.

We elect to implement option (2). We implement this by doing a reverse iteration over the nodes of the `Pipe` module and applying the following rules for each type of node:

* *call_module*. For module calls, we want to compute the gradient of each tensor input or module parameter with `requires_grad` with respect to the gradient of each tensor output that `requires_grad`. We can emit a call to a function `stage_backward(output_vals, dout)` that is a wrapper over `autograd.backward` (or autograd.grad). This wrapper handles unpacking/packing collection type inputs/outputs (like a pytree) and delegates to autograd.backward to compute and accumulates gradient values into the .grad attribute for each input and parameter value of this pipeline stage. `stage_backward` returns the gradients of the input values of the pipeline stage.
    * TODO: For gradient checkpointing, we should wrap the forward() invocation of the forward module with torch.utils.checkpoint. Then we can compute gradients in the same manner (TODO: is this right?)
    * TODO: is it good enough to dynamically figure out which tensor inputs require grad?
    * TODO: if the input tensors are values sent over the wire from the remote, do they have any attached grad_fn? If so, do we need to block the gradient somehow? Can we detach?
    * TODO: how to transmit saved output tensors on the same device without putting them through RPC? i.e. the data dependencies from the forward to the backward phase should just be passing a tensor reference, not going over RPC
        * Maybe just special-case this in the executor?
    * TODO: Zach mentioned that it is not always necessary to save the whole output tensor for computing the gradient. e.g. gradient of matmul does not require the output in the formulae for gradient of its inputs. Is there a way to call autograd.backward and only pass in the grad_fns and not the output tensors themselves? ask alban

* `call_function` + `operator.getitem`. This is used solely for the purpose of indexing into tuple outputs of stages if a stage has multiple outputs. In the backwards, the corresponding operation should be to rebuild the collection type for the purpose of passing it to `stage_backward`. We need to lazily build these collection types as we iterate in reverse order over the program
* placeholder - TODO: should we return gradients of pipeline inputs? Does anyone need this?

## Runtime

`PipelineDriver.py` contains the implementation for a single-driver multiple-follower runtime that interprets the abstract IR described above. The classes contained within this file are the following:

* `PipelineDriver` is the main user-facing entrypoint for using this runtime. The user feeds in the `Pipe` instance representing their model as well as the world size over which to run the pipeline. The user can optionally specify a non-trivial set of ranks over which to run the pipeline (i.e. not 0, 1, 2, ..., world_size-1). `PipelineDriver.run` is the main entry-point for actually running the model in a pipelined way. `run` takes as argument the initial arguments to the pipelined computation, the number of chunks to split the mini-batch into, optionally the batch dimension of each tensor input (assumed to be 0 if not specified), and optionally the initial environment, equivalent to the `initial_env` used in `fx.Interpreter`. Finally, a private argument for testing purposes is `_debug_mask_minibatches`, which, when `True`, implements the masking numeric correctness checking described below.

These classes are more geared toward being implementation details, but may be useful for overriding behavior:

* `PipeStageExecutor`, which is a class that is instantiated on the pipeline stage machines via an `rpc.remote` call. This object is instantiated for each pipeline stage submodule, and manages ownership of the module/parameters and invocation of that module.
  * `PipeStageExecutor.invoke` is an [async RPC function](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution) that does the following:
    * Populates a `WorkItem` structure that records the `args`/`kwargs` of the invocation and metadata such as the microbatch ID, how many arguments are remote values that must be waited on, and the `Future` object that will be signalled when the invocation is complete.
    * Puts the `WorkItem` onto a "waiting" runlist data structure (or the "running" runlist in the unlikely event that it has no input data dependencies) so that the scheduling system can keep track of this work
    * Initiates asynchronous transfers of remote arg/kwarg values. This is done via launching an async RPC to the `async_transfer` function on the local host. This function does a non-blocking call to an RPC on the remote, receives a future from that RPC, and installs a callback on that future that will update the `WorkItem` to inform it that one of its operands is ready, potentially also moving the `WorkItem` to the "ready" runlist
  * `PipeStageExecutor` holds a `worker_thread` attribute, which is a Python thread that acts as a sort of event loop. It will block waiting for work on the "ready" runlist and execute that work whenever all of its dependencies are ready. TODO: task selection here should be configurable
  * TODO: different execution schedules
  * TODO: backward execution
* `RemoteInterpreter` splits an input mini-batch into micro-batches and interprets the top-level `Pipe` graph, issuing `invoke` calls to the associated `PipeStageExecutors` to orchestrate execution of the program in a pipelined fashion. TODO: issue loss and backwards calls

# Testing Entrypoints

Testing entrypoints are the following:

* `local_test_forward.py` tests forward-only execution of a pipelined model with multiple processes on a single host. It should be launched via `launch_local_test_forward.sh`, which internally uses torchrun to spawn multiple processes and assign them all a unique rank

# A Note About Correctness Testing

Note that micro-batch splitting and reconstruction is not guaranteed to be bitwise-equivalent to running the same program on the full batch (see [here](https://pytorch.org/docs/master/notes/numerical_accuracy.html#batched-computations-or-slice-computations)). See also `exps/split_example.py`, which demonstrates this when constant `USE_WHOLE_BATCH` is set to `False`. A proposed way to get around this is, when testing for correctness, is to run the _full batch_ through the network for each micro-batch invocation and slice out the results from the full batch that correspond to each micro-batch, then cat those partial results together. This is demonstrated when `USE_WHOLE_BATCH` is `True`. This should guarantee numerical equivalence during testing while still exercising the micro-batch pipelining machinery.

# Training Loop API Considerations

During the training loop, we should focus on a few important elements:

* Data loader
* `forward`
  * This is already trivially handled by the current implementation. `Pipe` handles
    micro-batch computation of the model in the `forward` execution
* Loss
  * There are a few alternatives for API design here:
    1. We can preserve the training loop as-is and make it so that invoking a loss computation on the output
       of the forward() computation issues jobs on the last stage to compute the loss. We could use something
       similar to [DistributedLoss](https://github.com/facebookresearch/fairscale/blob/main/fairscale/experimental/nn/distributed_pipeline/loss.py#L16). An open question is how to represent the output of forward() in such a way
       that it can represent the forward() output _for all micro-batches_. This might look like a data structure
       that is a list of RRefs backed by async futures on the pipeline stages. Then, if we intercept computation
       on this loss object, we would issue `WorkItem`s for each operation for each micro-batch. However, this
       seems to degenerate down to a full-on tracing/single-coordinator system
    2. We can make it so that the pipeline API takes the loss as a funciton argument and essentially encapsulates the
       whole training loop. This is much easier, probably less flaky (in terms of not needing to build a whole tracing
       mechanism), but is not super Pythonic. It may facilitate implementing async pipeline parallelism in the future
* `backward`
  * There are similar considerations for `backward` as there are for `loss`. `backward` is an invocation on the
    scalar loss value that will need to schedule jobs in the backend.
* Optimizer
  * Requirements here are similar to loss and backwards, but the optimizer step happens only once for the
    whole mini-batch, so it may be the case that this can literally be the same as the normal optimizer
    (potentially using [DistributedOptimizer](https://pytorch.org/docs/master/distributed.optim.html)).

# Work Items

- [ ] Figure out how performance looks
- [ ] Loss execution in runtime
- [ ] backward() execution in runtime
- [ ] gradient checkpointing in runtime
- [ ] shared weights synchronization in runtime
- [ ] CUDA device placement
- [ ] Beta integration into repos like HuggingFace transformers

low-pri
- [ ] Shape-specialized tracing
- [ ] More APIs and algorithms for splitting in the front-end
- [ ] TRANSMIT synchronization type for leaf modules
- [ ] Yield running coroutines when I/O bound (e.g. when running a collective)
- [ ] Model side-effectful operations (resource semaphore, RNG state mutation, etc)
