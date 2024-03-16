#!/bin/bash

extra_args="$@"
seeds=(0 1 2)

for seed in "${seeds[@]}"
do
    python -u main.py seed=$seed model=pna data=mnist n_epochs=200 \
      model.graph_constructor.num_probe_features=0 model.gnn_backbone.dropout=0.2 \
      model.rev_edge_features=True \
      wandb.name=style_editing_mnist_dilation_pna_seed_${seed}_epoch_200_rev_edge_epoch_200_drop_0.2 \
      "$extra_args"
done
