# Graph Neural Networks for Learning Equivariant Representations of Neural Networks

Official implementation for
<pre>
<b>Graph Neural Networks for Learning Equivariant Representations of Neural Networks</b>
<a href="https://mkofinas.github.io/">Miltiadis Kofinas</a>*, <a href="https://bknyaz.github.io/">Boris Knyazev</a>, <a href="https://www.cyanogenoid.com/">Yan Zhang</a>, <a href="https://yunlu-chen.github.io/">Yunlu Chen</a>, <a href="https://gertjanburghouts.github.io/">Gertjan J. Burghouts</a>, <a href="https://egavves.com/">Efstratios Gavves</a>, <a href="https://www.ceessnoek.info/">Cees G. M. Snoek</a>, <a href="https://davzha.netlify.app/">David W. Zhang</a>*
<em>ICLR 2024</em>
<a href="https://arxiv.org/abs/2403.12143">https://arxiv.org/abs/2403.12143/</a>
*Joint first and last authors
</pre>

[![arXiv](https://img.shields.io/badge/arXiv-2403.12143-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2403.12143)
[![OpenReview](https://img.shields.io/badge/OpenReview-oO6FsMyDBt-b31b1b.svg)](https://openreview.net/forum?id=oO6FsMyDBt)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/neural_graphs_dark_transparent_bg.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/neural_graphs_light_transparent_bg.png">
  <img alt="Neural Graphs" src="assets/neural_graphs_light_transparent_bg.png">
</picture>

## Setup environment

To run the experiments, first create a clean virtual environment and install the requirements.

```bash
conda create -n neural-graphs python=3.9
conda activate neural-graphs
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg==2.3.0 pytorch-scatter -c pyg
pip install hydra-core einops opencv-python
```

Install the repo:

```bash
git clone https://https://github.com/mkofinas/neural-graphs.git
cd neural-graphs
pip install -e .
```

## Introduction Notebook

An introduction notebook for INR classification with **Neural Graphs**:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkofinas/neural-graphs/blob/main/notebooks/mnist-inr-classification.ipynb)
[![Jupyter](https://img.shields.io/static/v1.svg?logo=jupyter&label=Jupyter&message=View%20On%20Github&color=lightgreen)](notebooks/mnist-inr-classification.ipynb)

## Run experiments

To run a specific experiment, please follow the instructions in the README file within each experiment folder.
It provides full instructions and details for downloading the data and reproducing the results reported in the paper.

- INR classification: [`experiments/inr_classification`](experiments/inr_classification)
- INR style editing: [`experiments/style_editing`](experiments/style_editing)
- CNN generalization: [`experiments/cnn_generalization`](experiments/cnn_generalization)
- Learning to optimize (coming soon): [`experiments/learning_to_optimize`](experiments/learning_to_optimize)

## Datasets

### INR classification and style editing

For INR classification, we use MNIST and Fashion MNIST. **The datasets are available [here](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0).**

- [MNIST INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=mnist-inrs.zip)
- [Fashion MNIST INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=fmnist_inrs.zip)

For INR style editing, we use MNIST. **The dataset is available [here](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0).**

- [MNIST INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=mnist-inrs.zip)

### CNN generalization

For CNN generalization, we use the grayscale CIFAR-10 (CIFAR10-GS) from the
[_Small CNN Zoo_](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy)
dataset.
We also introduce *CNN Wild Park*, a dataset of CNNs with varying numbers of
layers, kernel sizes, activation functions, and residual connections between
arbitrary layers.

- [CIFAR10-GS](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz)
- CNN Wild Park (coming soon)

## Citation

If you find our work or this code to be useful in your own research, please consider citing the following paper:

```bib
@inproceedings{kofinas2024graph,
  title={{G}raph {N}eural {N}etworks for {L}earning {E}quivariant {R}epresentations of {N}eural {N}etworks},
  author={Kofinas, Miltiadis and Knyazev, Boris and Zhang, Yan and Chen, Yunlu and Burghouts,
    Gertjan J. and Gavves, Efstratios and Snoek, Cees G. M. and Zhang, David W.},
  booktitle = {12th International Conference on Learning Representations ({ICLR})},
  year={2024}
}
```

```bib
@inproceedings{zhang2023neural,
  title={{N}eural {N}etworks {A}re {G}raphs! {G}raph {N}eural {N}etworks for {E}quivariant {P}rocessing of {N}eural {N}etworks},
  author={Zhang, David W. and Kofinas, Miltiadis and Zhang, Yan and Chen, Yunlu and Burghouts, Gertjan J. and Snoek, Cees G. M.},
  booktitle = {Workshop on Topology, Algebra, and Geometry in Machine Learning (TAG-ML), ICML},
  year={2023}
}
```

## Acknowledgments

- This codebase started based on [github.com/AvivNavon/DWSNets](https://github.com/AvivNavon/DWSNets) and the DWSNet implementation is copied from there
- The NFN implementation is copied and slightly adapted from [github.com/AllanYangZhou/nfn](https://github.com/AllanYangZhou/nfn)
- We implemented the relational transformer in PyTorch following the JAX implementation at [github.com/CameronDiao/relational-transformer](https://github.com/CameronDiao/relational-transformer). Our implementation has some differences that we describe in the paper.

## Contributors

- [David W. Zhang](https://davzha.netlify.app/)
- [Miltiadis (Miltos) Kofinas](https://mkofinas.github.io/)
