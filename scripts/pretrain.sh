#!/bin/bash

# Pretraining script for GLAM model with specific configurations
python train.py  \
    --batch_size 48 \
    --learning_rate 4e-5 \
    --experiment_name bert_simclr_screen_aligned_word_aug_text_align_side_patch_s_no_pool_same_pos_attn \
    --devices 4 \
    --strategy 'ddp_find_unused_parameters_true' \
    --llm_type bert \
    --precision bf16-true \
    --accumulate_grad_batches 2 \
    --grad_ckpt \
    --weight_decay 0.2 \
    --warm_up 4000 \
    --emb_dim 512 \
    --max_steps 40000 \
    --linear_proj \
    --embed \
    --structural_cap \
    --slip \
    --img_size 518 \
    --crop_size 518 \
    --vit_grad_ckpt \
    --load_jpg \
    --raw_caption \
    --num_workers 8 \
    --mask_ratio 1.0 \
    --mask_meta 0.8 \
    --inter_view \
    --num_freeze_blocks 0 \
    --screen_only \
    --aligned_mlo \
    --align_orientation \
    --remove_text \
    --fixed_view \
    --patch_contrast \
    --aug_text \
    --heavy_aug \
    --mega_patch_size 4 \
    --mega_patch_stride 4 \
    --same_pos_contrast \
    --attn_pooler \










