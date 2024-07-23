#!/bin/bash

extra_args="$@"
seeds=(0 1 2)

for seed in "${seeds[@]}"
do
    python -u main.py seed=$seed model=rtransformer data=mnist n_epochs=200 \
      model.graph_constructor.num_probe_features=0 model.dropout=0.2 \
      model.graph_constructor.use_pos_embed=True model.modulate_v=True \
      wandb.name=inr_cls_mnist_rt_mod_pe_seed_${seed}_epoch_200_drop_0.2 \
      $extra_args
done
