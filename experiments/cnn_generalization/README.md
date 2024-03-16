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


### CNN Wild Park

Note: In the following commands, change the `--project` and the `--entity` according to
your WandB account, or change the corresponding `yaml` files.

#### Relational Transformer Hyperparameter Sweep

```sh
wandb sweep --project cnn-generalization --entity neural-graphs sweep_configs/sweep_cnn_park_transformer.yaml
```

#### Graph Network Hyperparameter Sweep

```sh
wandb sweep --project cnn-generalization --entity neural-graphs sweep_configs/sweep_cnn_park_gnn.yaml
```

#### StatNN Hyperparameter Sweep

```sh
wandb sweep --project cnn-generalization --entity neural-graphs sweep_configs/sweep_cnn_park_statnn.yaml
```

#### Train Relational Transformer on CNN Wild Park

```python
python -u main.py batch_size=128 data=cnn_park distributed=False eval_every=1000 \
  loss._target_=torch.nn.BCELoss model=rtransformer model.d_attn_hid=32 \
  model.d_edge=16 model.d_edge_hid=64 model.d_node=32 model.d_node_hid=64 \
  model.d_out_hid=64 model.dropout=0.2 model.pooling_method=cat model.pooling_layer_idx=last \
  model.n_heads=4 model.n_layers=3 n_epochs=20 seed=0
```
