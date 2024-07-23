#!/bin/bash

extra_args="$@"
seeds=(0 1 2)

for seed in "${seeds[@]}"
do
  python -u main.py seed=$seed model=pna data=zoo_cifar_nfn distributed=False \
    loss._target_=torch.nn.MSELoss batch_size=128 n_epochs=300 \
    data.train.augmentation=True data.linear_as_conv=False model.d_hid=256 \
    wandb.name=cnn_generalization_cnn_zoo_pna_seed_${seed}_epoch_300_mse_b_128_d_256_no_lac_aug $extra_args
done

