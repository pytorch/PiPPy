# PiPPy: Pipeline Parallelism for PyTorch

[**Why PiPPy?**](#why-pippy)
| [**Install guide**](#install)
| [**PiPPy Quickstart**](#pippy-quickstart)
| [**Future Work**](#future-work)

# Why PiPPy?

One of the most important techniques for advancing the state of the art in deep learning is scaling. Common techniques for scaling neural networks include _data parallelism_, _tensor/operation parallelism_, and _pipeline parallelism_. In many cases, pipeline parallelism in particular can be an effective technique for scaling, however it is often difficult to implement, requiring intrusive code changes to model code and difficult-to-implement runtime orchestration code. PiPPy aims to provide a toolkit that does said things automatically to allow high-productivity scaling of models.

# What is PiPPy?

The PiPPy project consists of a compiler and runtime stack for automated parallelism and scaling of PyTorch models. Currently, PiPPy focuses on _pipeline parallelism_, a technique in which the code of the model is partitioned and multiple _micro-batches_ execute different parts of the model code concurrently. To learn more about pipeline parallelism, see [this article](https://www.deepspeed.ai/tutorials/pipeline/).

![GPipe Schedule](https://i.imgur.com/eyUc947.png)
(Diagram from Huang, 2018)

PiPPy provides the following features that make pipeline parallelism easier:

* Automatic splitting of model code via PyTorch tracer. The goal is for the user to provide model code as-is to the system for parallelization, without having to make heavyweight modifications to make parallelism work.
* Related to the last point, PiPPy supports non-trivial topologies, including skip connections and tied weights/layers. PiPPy provides configurable behavior for tied weights, allowing for transmission across pipeline stages or replication and gradient synchronization.
* First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects). This is currently missing from the torchgpipe-based `torch.distributed.pipeline.sync.Pipe`.
* Composability with other parallelism schemes such as data parallelism or tensor splitting model parallelism (overall, known as "3d parallelism"). Currently, pipelining and data parallelism can be composed. Other compositions will be available in the future.
* Support for pipeline scheduling paradigms, including schedules like fill-drain (GPipe), 1F1B and interleaved 1F1B. More schedules will be added too.

For in-depth technical architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

# Install

PiPPy requires PyTorch version newer than 2.2.0.dev to work. To quickly install, for example, PyTorch nightly, run the following command from the same directory as this README:

```
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

You can also select the CUDA build of PyTorch if your system has NVIDIA GPUs, for example:

```
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html
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

PiPPy consists of two parts: a _compiler_ and a _runtime_. The compiler takes your model code, splits it up, and transforms it into a `Pipe`, which is a wrapper that describes the model at each pipeline stage and their data-flow relationship. The runtime executes the `Pipe` in parallel, handling things like micro-batch splitting, scheduling, communication, and gradient propagation, etc. We will cover the APIs for these concepts in this section.

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
        # 10 output classes
        self.output_proj = torch.nn.Linear(layer_dims[-1], 10)

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f'layer{i}')(x)

        return self.output_proj(x)


in_dim = 512
layer_dims = [512, 1024, 256]
mn = MyNetwork(in_dim, layer_dims).to(device)
```

This network is written as free-form Python code; it has not been modified for any specific parallelism technique.

Let us see our first usage of the `pippy.IR.Pipe` interface:

```python
from pippy.IR import annotate_split_points, Pipe, PipeSplitWrapper

annotate_split_points(mn, {'layer0': PipeSplitWrapper.SplitPoint.END,
                           'layer1': PipeSplitWrapper.SplitPoint.END})

batch_size = 32
example_input = torch.randn(batch_size, in_dim, device=device)
chunks = 4

pipe = Pipe.from_tracing(mn, chunks, example_args=(example_input,))
print(pipe)

"""
************************************* pipe *************************************
GraphModule(
  (submod_0): PipeStageModule(
    (L__self___layer0_mod_lin): Linear(in_features=512, out_features=512, bias=True)
  )
  (submod_1): PipeStageModule(
    (L__self___layer1_mod_lin): Linear(in_features=512, out_features=1024, bias=True)
  )
  (submod_2): PipeStageModule(
    (L__self___layer2_lin): Linear(in_features=1024, out_features=256, bias=True)
    (L__self___output_proj): Linear(in_features=256, out_features=10, bias=True)
  )
)

def forward(self, arg0):
    submod_0 = self.submod_0(arg0);  arg0 = None
    submod_1 = self.submod_1(submod_0);  submod_0 = None
    submod_2 = self.submod_2(submod_1);  submod_1 = None
    return [submod_2]
"""
```

So what's going on here? First, `Pipe.from_tracing` uses a PyTorch tracer to turn our model into a directed acyclic graph (DAG) representation. Then, it groups together the operations and parameters into _pipeline stages_. Stages are represented as `submod_N` submodules, where `N` is a natural number.

We used `annotate_split_points` to specify that the code should be split and the end of `layer0` and `layer1`. Our code has thus been split into _three_ pipeline stages. PiPPy also provides `SplitPoint.BEGINNING` if a user wants to split before certain annotation point.

While the `annotate_split_points` API gives users a way to specify the split points without modifying the model, PiPPy also provides an API for in-model annotation: `pipe_split()`. For details, you can read [this example](https://github.com/pytorch/PiPPy/blob/main/test/test_pipe.py).

This covers the basic usage of the `Pipe` API. For more information, see the documentation.

<!-- (TODO: link to docs when live) -->

## Using PipelineStage for Pipelined Execution

Given the above `Pipe` object, we can use one of the `PipelineStage` classes to execute our model in a pipelined fashion. First off, let us instantiate a `PipelineStage` instance:

```python
# We are using `torchrun` to run this example with multiple processes.
# `torchrun` defines two environment variables: `RANK` and `WORLD_SIZE`.
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Initialize distributed environment
import torch.distributed as dist
dist.init_process_group(rank=rank, world_size=world_size)

# Pipeline stage is our main pipeline runtime. It takes in the pipe object,
# the rank of this process, and the device.
from pippy.PipelineStage import PipelineStage
stage = PipelineStage(pipe, rank, device)
```

We can now run the pipeline by passing input to the first `PipelineStage`:

```python
# Input data
x = torch.randn(batch_size, in_dim, device=device)

# Run the pipeline with input `x`. Divide the batch into 4 micro-batches
# and run them in parallel on the pipeline
if rank == 0:
    stage(x)
elif rank == world_size - 1:
    output = stage()
else:
    stage()
```

Note that since we split our model into three stages, we must run this script with three workers. For this example, we will use `torchrun` to run multiple processes within a single machine for demonstration purposes. We can collect up all of the code blocks above into a file named [example.py](examples/basic/example.py) and then run it with `torchrun` like so:

```
torchrun --nproc_per_node=3 example.py
```

## Advanced: Pipeline Schedules

Pipeline parallel training of deep neural networks is _bidirectional_ since training requires running both forward- and back-propagation of the network. As a result, multiple items of work may be ready to run on a pipeline stage at a given time. The problem of selecting between these work items is known as _scheduling_, and a specific policy for selecting work-items is known as a _pipeline schedule_.

PiPPy provides both off-the-shelf pipeline schedules as described in the research literature as well as a programmable interface for creating new schedules. The schedules include:

* Fill-Drain. Fill-drain is a schedule that executes all forward microbatches before executing any backward microbatches. This is the "standard" schedule used in GPipe (Huang, 2018). Fill-drain scheduling can be used in PiPPy via the `PipelineDriverFillDrain` driver class. A diagram illustrating the fill-drain schedule is below.

<img src="https://i.imgur.com/eyUc947.png" alt="GPipe Schedule" width="800"/>
(Diagram from Huang, 2018)

* 1F1B (one forward, one backward) is a schedule that provides good hardware utilization as well as limits the amount of memory needed on a stage. At steady-state, a pipeline stage will alternate between processing forward and backward micro-batches. 1F1B was introduced in its asynchronous form in (Harlap, 2018) and in its synchronous form in (Narayanan, 2021). 1F1B scheduling can be used in PiPPy via the `PipelineDriver1F1B` driver class. A diagram illustrating the 1F1B schedule is below.

<img src="https://i.imgur.com/Voomtcd.png" alt="Synchronous 1F1B Schedule" width="800"/>
(Diagram from Narayanan, 2021)

* Interleaved 1F1B. Interleaved 1F1B is a variant of 1F1B that divides the program into smaller chunks and assigns multiple chunks per stage in a wrap-around fashion. Interleaving improves pipeline throughput with similar memory characteristics to 1F1B. Interleaved 1F1B was introduced by (Narayanan, 2021). Interleaved 1F1B can be using in PiPPy via the `PipelineDriverInterleaved1F1B` driver class.

<img src="https://i.imgur.com/ujCPZAU.png" alt="Interleaved 1F1B Schedule" width="800"/>
(Diagram from Narayanan, 2021)

# Future Work

Future work on PiPPy includes:

* Increasing automation. We aim to develop automated systems that can alleviate the burden of the user to specify things such as the batch dimension or pipeline split points. Automatic, optimal splitting of a program into balanced pipeline stages is an interesting research field with advances in the deep learning systems field (e.g. Zheng, 2022) and adjacent fields such as high-level synthesis for digital design (e.g. Zaretsky, 2007).
* Expanding to more forms of parallelism. PiPPy is our first foray into compiler-mediated distribution of PyTorch programs. We would like to explore expanding the analysis and partitioning capabilities enabled by a compiler stack to other forms of parallelism, including data parallelism, model parallelism, and MoE parallelism. Such automation is a rich area of research that we would like to contribute to.

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
* D. C. Zaretsky, G. Mittal, R. P. Dick and P. Banerjee, "Balanced Scheduling and Operation Chaining in High-Level Synthesis for FPGA Designs," 8th International Symposium on Quality Electronic Design (ISQED'07), 2007, pp. 595-601, doi: 10.1109/ISQED.2007.41.
* Lai, Z., Li, S., Tang, X., Ge, K., Liu, W., Duan, Y., Qiao, L., & Li, D. (2022). Merak: A Efficient Distributed DNN Training Framework with Automated 3D Parallelism for Giant Foundation Models. arXiv preprint arXiv:2206.04959.

## License
PiPPy is 3-clause BSD licensed, as found in the LICENSE file.

## Citing PiPPy

If you use PiPPy in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{pippy2022,
  author =       {James Reed, Pavel Belevich, Ke Wen, Howard Huang, Will Constable},
  title =        {PiPPy: Pipeline Parallelism for PyTorch},
  howpublished = {\url{https://github.com/pytorch/PiPPy}},
  year =         {2022}
}
```
