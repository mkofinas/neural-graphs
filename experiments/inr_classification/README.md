# INR classification

The following commands assume that you are executing them from the current directory `experiments/inr_classification`.
If you are in the root of the repository, please navigate to the `experiments/inr_classification` directory:

```sh
cd experiments/inr_classification
```

Activate the conda environment:

```sh
conda activate neural-graphs
```

## Setup

### Download the data

For INR classification, we use MNIST and Fashion MNIST.
The datasets are available [here](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0).

- [MNIST INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=mnist-inrs.zip)
- [Fashion MNIST INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=fmnist_inrs.zip)

Please download the data and place it in `dataset/mnist-inrs` and `dataset/fmnist_inrs`, respectively.
If you want to use a different path, please change the following commands
accordingly, or symlink your dataset path to the default ones.

#### MNIST

```sh
wget "https://www.dropbox.com/sh/56pakaxe58z29mq/AABrctdu2U65jGYr2WQRzmMna/mnist-inrs.zip?dl=0" -O mnist-inrs.zip &&
  mkdir -p dataset/mnist-inrs &&
  unzip -q mnist-inrs.zip -d dataset &&
  rm mnist-inrs.zip
```

#### Fashion MNIST

```sh
wget "https://www.dropbox.com/sh/56pakaxe58z29mq/AAAssoHq719OmSHSKKTiKKHGa/fmnist_inrs.zip?dl=0" -O fmnist_inrs.zip &&
  mkdir -p dataset/fmnist_inrs &&
  unzip -q fmnist_inrs.zip -d dataset &&
  rm fmnist_inrs.zip
```

### Data preprocessing

We have already performed the data preprocessing required for MNIST and Fashion
MNIST and provide the files within the repository. The preprocessing generates the
data splits and the dataset statistics. These correspond to the files
`dataset/mnist_splits.json` and `dataset/mnist_statistics.pth`
for MNIST, and `dataset/fmnist_splits.json` and `dataset/fmnist_statistics.pth`
for Fashion MNIST.

However, if you want to use different directories for your experiments, you have
to run the scripts that follow, or simply symlink your paths to the default ones.

#### MNIST

First, create the data split using:

```shell
python dataset/generate_mnist_data_splits.py \
  --data-path mnist-inrs --save-path . --name mnist_splits.json
```
This will create a json file `dataset/mnist_splits.json`.
**Note** that the `--data-path` and `--save-path` arguments should be set relatively
to the `dataset` directory.

Next, compute the dataset (INRs) statistics:
```shell
python dataset/compute_mnist_statistics.py \
  --data-path . --save-path . \
  --splits-path mnist_splits.json --statistics-path mnist_statistics.pth
```
This will create `dataset/mnist_statistics.pth` object.
Again, `--data-path` and `--save-path` should be set relatively to the `dataset`
directory.

#### Fashion MNIST

Fashion MNIST requires a slightly different preprocessing.
First, prepare the data splits using:

```shell
python dataset/preprocess_fmnist.py \
  --data-path fmnist_inrs/splits.json --save-path . --name fmnist_splits.json
```

Next, compute the dataset statistics:
```shell
python dataset/compute_fmnist_statistics.py \
  --data-path . --save-path . \
  --splits-path fmnist_splits.json --statistics-path fmnist_statistics.pth
```
This will create `dataset/fmnist_statistics.pth` object.


## Run the experiment

Now for the fun part! :rocket:
To train and evaluate a __Neural Graph Transformer__ (NG-T) model on the MNIST dataset, run the following command:

```shell
python main.py model=rtransformer data=mnist
```

Make sure to check the model configuration in `configs/model/rtransformer.yaml`
and the data configuration in `configs/data/mnist.yaml`.
If you used different paths for the data, you can either overwrite the default
paths in `configs/data/mnist.yaml` or pass the paths as arguments to the command:

```shell
python main.py model=rtransformer data=mnist \
  data.dataset_dir=<your-dataset-dir> data.splits_path=<your-splits-path> \
  data.statistics_path=<your-statistics-path>
```

Training a different model or using a different dataset is as simple as changing
the `model` and `data` arguments!
For example, you can train and evaluate a __Neural Graph Graph Neural Network__ (NG-GNN)
on Fashion MNIST using the following command:

```shell
python main.py model=pna data=fmnist
```

### Run experiments with scripts

You can also run the experiments using the scripts provided in the `scripts` directory.
For example, to train and evaluate a __Neural Graph Transformer__ (NG-T) model on the MNIST dataset, run the following command:

```sh
./scripts/mnist_cls_rt.sh
```
This script will run the experiment for 3 different seeds.
