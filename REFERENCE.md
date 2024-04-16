# Advanced: Pipeline Schedules

Pipeline parallel training of deep neural networks is _bidirectional_ since training requires running both forward- and back-propagation of the network. As a result, multiple items of work may be ready to run on a pipeline stage at a given time. The problem of selecting between these work items is known as _scheduling_, and a specific policy for selecting work-items is known as a _pipeline schedule_.

PiPPy provides both off-the-shelf pipeline schedules as described in the research literature as well as a programmable interface for creating new schedules. The schedules include:

* Fill-Drain. Fill-drain is a schedule that executes all forward microbatches before executing any backward microbatches. This is the "standard" schedule used in GPipe (Huang, 2018).

* 1F1B (one forward, one backward) is a schedule that provides good hardware utilization as well as limits the amount of memory needed on a stage. At steady-state, a pipeline stage will alternate between processing forward and backward micro-batches. 1F1B was introduced in its asynchronous form in (Harlap, 2018) and in its synchronous form in (Narayanan, 2021).

* Interleaved 1F1B. Interleaved 1F1B is a variant of 1F1B that divides the program into smaller chunks and assigns multiple chunks per stage in a wrap-around fashion. Interleaving improves pipeline throughput with similar memory characteristics to 1F1B. Interleaved 1F1B was introduced by (Narayanan, 2021).

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
