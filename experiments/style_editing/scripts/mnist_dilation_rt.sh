#!/bin/bash

extra_args="$@"
seeds=(0 1 2)

for seed in "${seeds[@]}"
do
    python -u main.py seed=$seed model=rtransformer data=mnist n_epochs=200 \
      model.graph_constructor.num_probe_features=0 model.d_node=64 model.d_edge=32 \
      model.dropout=0.3 model.graph_constructor.use_pos_embed=True \
      model.modulate_v=True model.rev_edge_features=True \
      wandb.name=style_editing_mnist_dilation_rt_seed_${seed}_hid_64_epoch_200_rev_edge_epoch_200_drop_0.3 \
      "$extra_args"
done
