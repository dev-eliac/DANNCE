#!/bin/bash

# You can change the seed if needed
random_seed=16

echo "Running DANNCE (Transformer) for OfficeHome: Art target..."

# Note: We need to point python to the module correctly.
# Assuming we run this script FROM THE PROJECT ROOT (works/DANNCE):
python3 -m src.main \
    --dataset=OfficeHome \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=officehome/dannce-transformer/art-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --features_lr=1e-5 \
    --classifier_lr=1e-4 \
    --domain_adversary_lr=1e-4 \
    --adversarial_examples_lr=1e-3 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-3 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=art \
    --use_original_train_set \
    --entropy \
    --num_epochs=30

echo "Done."