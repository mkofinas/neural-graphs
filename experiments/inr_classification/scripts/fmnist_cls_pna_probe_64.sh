#!/bin/bash

extra_args="$@"
seeds=(0 1 2)

for seed in "${seeds[@]}"
do
    python -u main.py seed=$seed model=pna data=fmnist n_epochs=200 \
      data.train.augmentation=True model.graph_constructor.num_probe_features=64 \
      model.gnn_backbone.dropout=0.2 model.graph_constructor.use_pos_embed=True \
      model.modulate_v=True model.rev_edge_features=True \
      wandb.name=inr_cls_fmnist_pna_probe_64_mod_pe_seed_${seed}_epoch_200_drop_0.2 \
      $extra_args
done
