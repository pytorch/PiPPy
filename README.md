# [experimental] PiPPy: Pipeline Parallelism for PyTorch

[**Why PiPPy?**](#why-pippy)
| [**Install guide**](#install)
| [**PiPPy Quickstart**](#pippy-quickstart)
| [**Future Work**](#future-work)
| [**References**](#references)

# Why PiPPy?

One of the most important techniques for advancing the state of the art in deep learning is scaling. Common techniques for scaling neural networks include _data parallelism_, _tensor/model parallelism_, and _pipeline parallelism_. In many cases, pipeline parallelism in particular can be an effective technique for scaling, however it is often difficult to implement, requiring intrusive code changes to model code and difficult-to-implement runtime orchestration code. PiPPy aims to provide a toolkit that does said things automatically to allow high-productivity scaling of models.

# What is PiPPy?

The PiPPy project consists of a compiler and runtime stack for automated parallelism and scaling of PyTorch models. Currently, PiPPy focuses on _pipeline parallelism_, a technique in which the code of the model is partitioned and multiple _micro-batches_ execute different parts of the model code concurrently. To learn more about pipeline parallelism, see [this article](https://www.deepspeed.ai/tutorials/pipeline/).

![GPipe Schedule](https://i.imgur.com/eyUc947.png)

PiPPy provides the following features that make pipeline parallelism easier:

* Automatic splitting of model code via `torch.fx`. The goal is for the user to provide model code as-is to the system for parallelization, without having to make heavyweight modifications to make parallelism work.
* Related to the last point, PiPPY supports non-trivial topologies, including skip connections and tied weights/layers. Pippy provides configurable behavior for tied weights, allowing for transmission across pipeline stages or replication and gradient synchronization.
* First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects). This is currently missing from the torchgpipe-based `torch.distributed.pipeline.sync.Pipe`.
* Composability with other parallelism schemes such as data parallelism or tensor splitting model parallelism (overall, known as "3d parallelism"). Currently, pipelining and data parallelism can be composed. Other compositions will be available in the future.
* Support for pipeline scheduling paradigms, including static schedules like fill-drain (GPipe), 1f1b, interleaved 1f1b and dynamic schedules like lookahead or registers/back-pressure.

For in-depth technical architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

# Install

PiPPy currently requires the PyTorch nightly release to work. To quickly install PyTorch nightly, run the following command from the same directory as this README:

```
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

To install PiPPy from source, run the following command in the same directory as this README:

```
python setup.py install
```

To expose PiPPy for development such that changes to this repo are reflected in the imported package, run:

```
python setup.py develop
```

# PiPPy Quickstart

PiPPy consists of two parts: a _compiler_ and a _runtime_. The compiler takes your model code, splits it up, and transforms it into a `Pipe`, which is a wrapper that describes how to execute the model in pipeline parallelism. The runtime executes the `Pipe` in parallel, handling things like micro-batch splitting and gradient propagation/syncing. We will cover the APIs for these concepts in this section.

## Splitting a Model with Pipe

To see how we can split a model into a pipeline, let's first take an example trivial neural network:

```python
import torch

class MyNetworkBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


class MyNetwork(torch.nn.Module):
    def __init__(self, in_dim, layer_dims):
        super().__init__()

        prev_dim = in_dim
        for i, dim in enumerate(layer_dims):
            setattr(self, f'layer{i}', MyNetworkBlock(prev_dim, dim))
            prev_dim = dim

        self.num_layers = len(layer_dims)

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f'layer{i}')(x)

        return x

mn = MyNetwork(512, [512, 1024, 256])
```

This network is written as free-form Python code; it has not been modified for any specific parallelism technique.

Let us see our first usage of the `pippy.IR.Pipe` interface:

```python
from pippy.IR import Pipe

pipe = Pipe.from_tracing(mn)
print(pipe)
"""
GraphModule(
  (submod_0): GraphModule(
    (layer0_lin): Linear(in_features=512, out_features=512, bias=True)
    (layer1_lin): Linear(in_features=512, out_features=1024, bias=True)
    (layer2_lin): Linear(in_features=1024, out_features=256, bias=True)
  )
)



def forward(self, x):
    submod_0 = self.submod_0(x);  x = None
    return submod_0
"""
print(pipe.split_gm.submod_0)
"""
def forward(self, x):
    layer0_lin = self.layer0_lin(x);  x = None
    relu = torch.relu(layer0_lin);  layer0_lin = None
    layer1_lin = self.layer1_lin(relu);  relu = None
    relu_1 = torch.relu(layer1_lin);  layer1_lin = None
    layer2_lin = self.layer2_lin(relu_1);  relu_1 = None
    relu_2 = torch.relu(layer2_lin);  layer2_lin = None
    return relu_2
"""
```

So what's going on here? First, `Pipe.from_tracing` uses `torch.fx` symbolic tracing to turn our model into a directed acyclic graph (DAG) representation. Then, it groups together the operations and parameters into _pipeline stages_. Stages are represented as `submod_N` submodules, where `N` is a natural number.

The above code groups together our model's operators and parameters into only one stage, since we have not specified a splitting policy. Let us add a custom splitting policy:

```python
from pippy.IR import annotate_split_points, PipeSplitWrapper

annotate_split_points(mn, {'layer0': PipeSplitWrapper.SplitPoint.END,
                           'layer1': PipeSplitWrapper.SplitPoint.END})

pipe = Pipe.from_tracing(mn)

print(' pipe '.center(80, '*'))
print(pipe)
"""
************************************* pipe *************************************
GraphModule(
  (submod_0): GraphModule(
    (layer0_mod_lin): Linear(in_features=512, out_features=512, bias=True)
  )
  (submod_1): GraphModule(
    (layer1_mod_lin): Linear(in_features=512, out_features=1024, bias=True)
  )
  (submod_2): GraphModule(
    (layer2_lin): Linear(in_features=1024, out_features=256, bias=True)
  )
)



def forward(self, x):
    submod_0 = self.submod_0(x);  x = None
    submod_1 = self.submod_1(submod_0);  submod_0 = None
    submod_2 = self.submod_2(submod_1);  submod_1 = None
    return submod_2
"""

print(' submod0 '.center(80, '*'))
print(pipe.split_gm.submod_0)
"""
*********************************** submod0 ************************************
GraphModule(
  (layer0_mod_lin): Linear(in_features=512, out_features=512, bias=True)
)



def forward(self, x):
    layer0_mod_lin = self.layer0_mod_lin(x);  x = None
    relu = torch.relu(layer0_mod_lin);  layer0_mod_lin = None
    return relu
"""

print(' submod1 '.center(80, '*'))
print(pipe.split_gm.submod_1)
"""
*********************************** submod1 ************************************
GraphModule(
  (layer1_mod_lin): Linear(in_features=512, out_features=1024, bias=True)
)



def forward(self, relu):
    layer1_mod_lin = self.layer1_mod_lin(relu);  relu = None
    relu_1 = torch.relu(layer1_mod_lin);  layer1_mod_lin = None
    return relu_1
"""

print(' submod2 '.center(80, '*'))
print(pipe.split_gm.submod_2)
"""
*********************************** submod2 ************************************
GraphModule(
  (layer2_lin): Linear(in_features=1024, out_features=256, bias=True)
)



def forward(self, relu_1):
    layer2_lin = self.layer2_lin(relu_1);  relu_1 = None
    relu = torch.relu(layer2_lin);  layer2_lin = None
    return relu
"""
```

Our code has now been split into _three_ pipeline stages. We used `annotate_split_points` to specify that the code should be split and the end of `layer0` and `layer1`.

This covers the basic usage of the `Pipe` API. For more information, see the documentation.

<!-- (TODO: link to docs when live) -->

## Using PipelineDriver for Pipelined Execution

Given the above `Pipe` object, we can use one of the `PipelineDriver` classes to execute our model in a pipelined fashion. First off, let us instantiate a `PipelineDriverFillDrain` instance:

```python
# To run a distributed training job, we must launch the script in multiple
# different processes. We are using `torchrun` to do so in this example.
# `torchrun` defines two environment variables: `LOCAL_RANK` and `WORLD_SIZE`,
# which represent the index of this process within the set of processes and
# the total number of processes, respectively.
#
# To learn more about `torchrun`, see
# https://pytorch.org/docs/stable/elastic/run.html
import os
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ['WORLD_SIZE'])

# PiPPy uses the PyTorch RPC interface. To use RPC, we must call `init_rpc`
# and inform the RPC framework of this process's rank and the total world
# size. We can directly pass values `torchrun` provided.`
#
# To learn more about the PyTorch RPC framework, see
# https://pytorch.org/docs/stable/rpc.html
import torch.distributed.rpc as rpc
rpc.init_rpc(f'worker{local_rank}', rank=local_rank, world_size=world_size)

# PiPPy relies on the concept of a "driver" process. The driver process
# should be a single process within the RPC group that instantiates the
# PipelineDriver and issues commands on that object. The other processes
# in the RPC group will receive commands from this process and execute
# the pipeline stages
if local_rank == 0:
    # We are going to use the PipelineDriverFillDrain class. This class
    # provides an interface for executing the `Pipe` in a style similar
    # to the GPipe fill-drain schedule. To learn more about GPipe and
    # the fill-drain schedule, see https://arxiv.org/abs/1811.06965
    from pippy.PipelineDriver import PipelineDriverFillDrain
    from pippy.microbatch import TensorChunkSpec

    # Pipelining relies on _micro-batching_--that is--the process of
    # dividing the program's input data into smaller chunks and
    # feeding those chunks through the pipeline sequentially. Doing
    # this requires that the data and operations be _separable_, i.e.
    # there should be at least one dimension along which data can be
    # split such that the program does not have interactions across
    # this dimension. PiPPy provides `chunk_spec` arguments for this
    # purpose, to specify the batch dimension for tensors in each of
    # the args, kwargs, and outputs. The structure of the `chunk_spec`s
    # should mirror that of the data type. Here, the program has a
    # single tensor input and single tensor output, so we specify
    # a single `TensorChunkSpec` instance indicating dimension 0
    # for args[0] and the output value.
    args_chunk_spec = (TensorChunkSpec(0),)
    kwargs_chunk_spec = {}
    output_chunk_spec = TensorChunkSpec(0)

    # Finally, we instantiate the PipelineDriver. We pass in the pipe,
    # chunk specs, and world size, and the constructor will distribute
    # our code to the processes in the RPC group. `driver` is an object
    # we can invoke to run the pipeline.
    driver = PipelineDriverFillDrain(
        pipe, args_chunk_spec=args_chunk_spec, kwargs_chunk_spec=kwargs_chunk_spec,
        output_chunk_spec=output_chunk_spec, world_size=world_size)

    # <following code goes here>

rpc.shutdown()
```

Note that our script must now be replicated across multiple workers. For this example, we will use `torchrun` to run multiple processes within a single machine for demonstration purposes. So, if you've named the above code `example.py`, the `torchrun` invocation should look like:

```
torchrun --nproc_per_node=3 example.py
```

Note that we have launched 3 processes, as we have 3 pipeline stages.

We can now run the pipeline by using the `PipelineDriver.run` method (make sure to add this code in the `<bracketed>` area above):

```python
    # Run the pipeline with input `x`. Divide the batch into 64 micro-batches
    # and run them in parallel on the pipeline
    output = driver.run(64, x)

    # Run the original code and get the output for comparison
    reference_output = mn(x)

    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)
```

We can see that we can now execute our model in a pipelined fashion and get the same numeric outputs.

## Forward vs. Forward-loss-backward

The above example demonstrated only pipelining the `forward()` computation, for example for the purposes of model evaluation. We can extend the example to include the loss and back-propagation computation for the purposes of model training. Let us replace the code under the `if local_rank == 0:` block in the example:

```python
if local_rank == 0:
    from pippy.PipelineDriver import PipelineDriverFillDrain
    from pippy.microbatch import TensorChunkSpec

    # LossWrapper is a convenient base class you can use to compose your model
    # with the desired loss function for the purpose of pipeline parallel training.
    # Since the loss is executed as part of the pipeline, it cannot reside in the
    # training loop, so you must embed it like this
    from pippy.IR import LossWrapper
    class ModelLossWrapper(LossWrapper):
        def forward(self, x, target):
            return self.loss_fn(self.module(x), target)

    # TODO: mean reduction
    loss_wrapper = ModelLossWrapper(module=mn, loss_fn=torch.nn.MSELoss(reduction='sum'))

    # Instantiate the `Pipe` similarly to before, but with two differences:
    #   1) We pass in the `loss_wrapper` module to include the loss in the
    #      computation
    #   2) We specify `output_loss_value_spec`. This is a data structure
    #      that should mimic the structure of the output of LossWrapper
    #      and has a True value in the position where the loss value will
    #      be. Since LossWrapper returns just the loss, we just pass True
    pipe = Pipe.from_tracing(loss_wrapper, output_loss_value_spec=True)

    # We now have two args: `x` and `target`, so specify batch dimension
    # for both.
    args_chunk_spec = (TensorChunkSpec(0), TensorChunkSpec(0))
    kwargs_chunk_spec = {}
    # The output is now a `loss` value, which is a scalar tensor.
    # PiPPy's default is to concatenate outputs, but that will not
    # work with a scalar tensor. So we use a CustomReducer instead
    # to merge together the partial loss values.
    from pippy.microbatch import CustomReducer
    output_chunk_spec = CustomReducer(0.0, lambda a, b: a + b)

    # Instantiate the driver as usual.
    driver = PipelineDriverFillDrain(
        pipe, args_chunk_spec=args_chunk_spec, kwargs_chunk_spec=kwargs_chunk_spec,
        output_chunk_spec=output_chunk_spec, world_size=world_size)
```

The comments describe the new components that have been added to enable training. We can print out the new `pipe` to see the loss and backward stages:

```python
  print(pipe)

  """
  def forward(self, x, target):
      submod_0 = self.submod_0(x)
      submod_1 = self.submod_1(submod_0)
      submod_2 = self.submod_2(submod_1, target)
      stage_backward = pippy_IR_stage_backward(stage_output = (submod_2,), output_grads = (None,), input_values = [submod_1, target], outputs_with_grads_idxs = [0], stage_info = 'stage_backward for stage %submod_2 : [#users=2] = call_module[target=submod_2](args = (%submod_1, %target), kwargs = {})');  target = None
      getitem = stage_backward[0]
      getitem_1 = stage_backward[1];  stage_backward = None
      getitem_2 = getitem[0]
      getitem_3 = getitem[1];  getitem = None
      stage_backward_1 = pippy_IR_stage_backward(stage_output = (submod_1,), output_grads = (getitem_2,), input_values = [submod_0], outputs_with_grads_idxs = [0], stage_info = 'stage_backward_1 for stage %submod_1 : [#users=3] = call_module[target=submod_1](args = (%submod_0,), kwargs = {})');  submod_1 = getitem_2 = None
      getitem_4 = stage_backward_1[0]
      getitem_5 = stage_backward_1[1];  stage_backward_1 = None
      getitem_6 = getitem_4[0];  getitem_4 = None
      stage_backward_2 = pippy_IR_stage_backward(stage_output = (submod_0,), output_grads = (getitem_6,), input_values = [x], outputs_with_grads_idxs = [0], stage_info = 'stage_backward_2 for stage %submod_0 : [#users=3] = call_module[target=submod_0](args = (%x,), kwargs = {})');  submod_0 = getitem_6 = x = None
      getitem_7 = stage_backward_2[0]
      getitem_8 = stage_backward_2[1];  stage_backward_2 = None
      getitem_9 = getitem_7[0];  getitem_7 = None
      sync_barrier = pippy_IR_sync_barrier(submod_2, [getitem_1, getitem_5, getitem_8]);  submod_2 = getitem_1 = getitem_5 = getitem_8 = None
      return sync_barrier
  """
```

As before, we can now call the `driver` object to execute the pipeline; However this time, the forward, loss, and backward passes will all be executed:

```python
    x = torch.randn(512, 512)
    target = torch.randn(512, 256)

    # note the additional `target` argument, as the module we're running is
    # ModelLossWrapper
    output = driver.run(64, x, target)

    reference_output = loss_wrapper(x, target)

    # Compare numerics of pipeline and original model
    torch.testing.assert_close(output, reference_output)
```

<!-- TODO: check gradients -->
<!-- TODO: optimizer & LR scheduler -->

## PiPPy on CUDA

<!-- TODO -->

## PiPPy + Data Parallelism

<!-- TODO -->

## Advanced: Pipeline Schedules

Pipeline parallel training of deep neural networks is _bidirectional_ since training requires running both forward- and back-propagation of the network. As a result, multiple items of work may be ready to run on a pipeline stage at a given time. The problem of selecting between these work items is known as _scheduling_, and a specific policy for selecting work-items is known as a _pipeline schedule_.

PiPPy provides both off-the-shelf pipeline schedules as described in the research literature as well as a programmable interface for creating new schedules. The schedules include:

* Fill-Drain. Fill-drain is a schedule that executes all forward microbatches before executing any backward microbatches. This is the "standard" schedule used in GPipe (Huang, 2018). Fill-drain scheduling can be used in PiPPy via the `PipelineDriverFillDrain` driver class. A diagram illustrating the fill-drain schedule is below.

<img src="https://i.imgur.com/eyUc947.png" alt="GPipe Schedule" width="800"/>

* 1F1B (one forward, one backward) is a schedule that provides good hardware utilization as well as limits the amount of memory neeed on a stage. At steady-state, a pipeline stage will alternate between processing forward and backward micro-batches. 1F1B was introduced in its asynchronous form in (Harlap, 2018) and in its synchronous form in (Narayanan, 2021). 1F1B scheduling can be used in PiPPy via the `PipelineDriver1F1B` driver class. A diagram illustrating the 1F1B schedule is below.

<img src="https://i.imgur.com/Voomtcd.png" alt="Synchronous 1F1B Schedule" width="800"/>

* Interleaved 1F1B. Interleaved 1F1B is a variant of 1F1B that divides the program into smaller chunks and assigns multiple chunks per stage in a wrap-around fashion. Interleaving improves pipeline throughput with similar memory characteristics to 1F1B. Interleaved 1F1B was introduced by (Narayanan, 2021). Interleaved 1F1B can be using in PiPPy via the `PipelineDriverInterleaved1F1B` driver class.

<img src="https://i.imgur.com/ujCPZAU.png" alt="Interleaved 1F1B Schedule" width="800"/>

# Future Work

# References

* Chi-Chung Chen, Chia-Lin Yang, & Hsiang-Yun Cheng (2018). Efficient and Robust Parallel DNN Training through Model Parallelism on Multi-GPU Platform. CoRR, abs/1809.02839.
* Geng, J., Li, D., & Wang, S. (2019). ElasticPipe: An Efficient and Dynamic Model-Parallel Solution to DNN Training. In Proceedings of the 10th Workshop on Scientific Cloud Computing (pp. 5–9). Association for Computing Machinery.
* Lei Guan and Wotao Yin and Dongsheng Li and Xicheng Lu (2019). XPipe: Efficient Pipeline Model Parallelism for Multi-GPU DNN Training. CoRR, abs/1911.04610.
* Aaron Harlap and Deepak Narayanan and Amar Phanishayee and Vivek Seshadri and Nikhil R. Devanur and Gregory R. Ganger and Phillip B. Gibbons (2018). PipeDream: Fast and Efficient Pipeline Parallel DNN Training. CoRR, abs/1806.03377.
*Yanping Huang and Yonglong Cheng and Dehao Chen and HyoukJoong Lee and Jiquan Ngiam and Quoc V. Le and Zhifeng Chen (2018). GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. CoRR, abs/1811.06965.
* Chiheon Kim and Heungsub Lee and Myungryong Jeong and Woonhyuk Baek and Boogeon Yoon and Ildoo Kim and Sungbin Lim and Sungwoong Kim (2020). torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models. CoRR, abs/2004.09910.
* Atli Kosson and Vitaliy Chiley and Abhinav Venigalla and Joel Hestness and Urs Köster (2020). Pipelined Backpropagation at Scale: Training Large Models without Batches. CoRR, abs/2003.11666.
* Deepak Narayanan and Amar Phanishayee and Kaiyu Shi and Xie Chen and Matei Zaharia (2020). Memory-Efficient Pipeline-Parallel DNN Training. CoRR, abs/2006.09503.
* Deepak Narayanan and Mohammad Shoeybi and Jared Casper and Patrick LeGresley and Mostofa Patwary and Vijay Korthikanti and Dmitri Vainbrand and Prethvi Kashinkunti and Julie Bernauer and Bryan Catanzaro and Amar Phanishayee and Matei Zaharia (2021). Efficient Large-Scale Language Model Training on GPU Clusters. CoRR, abs/2104.04473.
* Petrowski, A., Dreyfus, G., & Girault, C. (1993). Performance analysis of a pipelined backpropagation parallel algorithm. IEEE Transactions on Neural Networks, 4(6), 970-981.
* Bowen Yang and Jian Zhang and Jonathan Li and Christopher Ré and Christopher R. Aberger and Christopher De Sa (2019). PipeMare: Asynchronous Pipeline Parallel DNN Training. CoRR, abs/1910.05124.
* Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Joseph E. Gonzalez, & Ion Stoica (2022). Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning. CoRR, abs/2201.12023.

## License
PiPPy is 3-clause BSD licensed, as found in the LICENSE file.

## Citing PiPPy

If you use PiPPy in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{pippy2022,
  author =       {James Reed, Pavel Belevich, Ke Wen},
  title =        {PiPPy: Pipeline Parallelism for PyTorch},
  howpublished = {\url{https://github.com/pytorch/PiPPy}},
  year =         {2022}
}
```
