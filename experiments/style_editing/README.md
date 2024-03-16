# INR style editing

The following commands assume that you are executing them from the current directory `experiments/style_editing`.
If you are in the root of the repository, please navigate to the `experiments/style_editing` directory:

```sh
cd experiments/style_editing
```

Activate the conda environment:

```sh
conda activate neural-graphs
```

## Setup

Follow the directions from the [INR classification
experiment](../inr_classification#setup) to download the data and
preprocess it. The default
dataset directory is `dataset` and it is shared with the `inr_classification` experiment.

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

Training a different model is as simple as changing the `model` argument!
For example, you can train and evaluate a __Neural Graph Graph Neural Network__ (NG-GNN)
on MNIST using the following command:

```shell
python main.py model=pna data=mnist
```

### Run experiments with scripts

You can also run the experiments using the scripts provided in the `scripts` directory.
For example, to train and evaluate a __Neural Graph Transformer__ (NG-T) model on the MNIST dataset, run the following command:

```sh
./scripts/mnist_dilation_rt.sh
```
This script will run the experiment for 3 different seeds.
