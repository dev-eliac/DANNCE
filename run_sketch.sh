#!/bin/bash

# You can change the seed if needed, keeping 16 to match your example
random_seed=16

echo "Running DANNCE (ResNeXt-50) for Sketch target..."

python3 -m src.main \
    --model resnext50_32x4d \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=pacs/dannce-resnext50/sketch-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-3 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-3 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=sketch \
    --use_original_train_set \
    --entropy

echo "Done."
