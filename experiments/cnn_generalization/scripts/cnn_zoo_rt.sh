#!/bin/bash

extra_args="$@"
seeds=(0 1 2)

for seed in "${seeds[@]}"
do
  python -u main.py seed=$seed model=rtransformer data=zoo_cifar_nfn distributed=False \
    loss._target_=torch.nn.BCELoss batch_size=192 n_epochs=300 \
    model.d_node=128 model.d_edge=64 model.d_attn_hid=256 model.d_node_hid=256 \
    model.d_edge_hid=128 model.d_out_hid=256 \
    wandb.name=cnn_generalization_cnn_zoo_rt_seed_${seed}_epoch_300_bce_double_b_192 $extra_args
done
