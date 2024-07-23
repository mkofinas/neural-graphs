## CNN generalization

### NFN CNN Zoo data

This experiment follows [NFN](https://arxiv.org/abs/2302.14040).
Download the
[CIFAR10](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz)
data  (originally from [Unterthiner et al,
2020](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy))
into `./dataset`, and extract them. Change `data_path` in
`./configs/data/zoo_cifar_nfn.yaml` if you want to store the data somewhere else.

Options for `data`:
- `zoo_cifar_nfn`: NFN CNN Zoo (CIFAR) dataset
<!-- - `zoo_svhn` : CNN Zoo (SVHN) dataset -->


#### Run experiments with scripts

You can run the experiments using the scripts provided in the `scripts` directory.
For example, to train and evaluate a __Neural Graph Transformer__ (NG-T) model on the CNN Zoo dataset, run the following command:

```sh
./scripts/cnn_zoo_rt.sh
```
This script will run the experiment for 3 different seeds.

### CNN Wild Park

[![CNN Wild Park](https://img.shields.io/badge/Zenodo-CNN%20Wild%20Park-blue?logo=zenodo)](https://doi.org/10.5281/zenodo.12797219)

Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.12797219) and extract it into `./dataset`.

#### Run experiments with scripts

You can run the experiments using the scripts provided in the `scripts` directory.
For example, to train and evaluate a __Neural Graph Transformer__ (NG-T) model on the CNN Wild Park dataset, run the following command:

```sh
./scripts/cnn_zoo_rt.sh
```
This script will run the experiment for 3 different seeds.

#### Hyperparameter Sweep

We also provide sweep configs for NG-T, NG-GNN, and StatNN in the `sweep_configs` directory.
In the following commands, change the `--project` and the `--entity` according to
your WandB account, or change the corresponding `yaml` files.

__NG-T__:
```sh
wandb sweep --project cnn-generalization --entity neural-graphs sweep_configs/sweep_cnn_park_transformer.yaml
```

__NG-GNN__:
```sh
wandb sweep --project cnn-generalization --entity neural-graphs sweep_configs/sweep_cnn_park_gnn.yaml
```

__StatNN__:
```sh
wandb sweep --project cnn-generalization --entity neural-graphs sweep_configs/sweep_cnn_park_statnn.yaml
```
