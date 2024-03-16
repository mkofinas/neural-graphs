#!/bin/bash

extra_args="$@"
seeds=(0 1 2)

for seed in "${seeds[@]}"
do
    python -u main.py seed=$seed model=rtransformer data=cnn_park distributed=False \
      eval_every=1000 n_epochs=20 batch_size=128 loss._target_=torch.nn.BCELoss \
      model.d_node=32 model.d_node_hid=64 model.d_edge=16 model.d_edge_hid=64 model.d_attn_hid=32 \
      model.d_out_hid=64 model.n_heads=4 model.n_layers=3 model.dropout=0.2 \
      model.pooling_method=cat model.pooling_layer_idx=last \
      wandb.name=cnn_generalization_cnn_park_rt_seed_${seed}_epoch_20_drop_0.2_bce \
      "$extra_args"
done
