#!/bin/bash

extra_args="$@"
seeds=(0 1 2)

for seed in "${seeds[@]}"
do
    python -u main.py seed=$seed model=pna data=cnn_park distributed=False \
      eval_every=1000 n_epochs=20 batch_size=128 loss._target_=torch.nn.BCELoss \
      model.d_hid=64 model.pooling_method=cat model.pooling_layer_idx=last  model.use_act_embed=False\
      wandb.name=cnn_generalization_cnn_park_pna_seed_${seed}_epoch_20_no_act_bce \
      "$extra_args"
done
