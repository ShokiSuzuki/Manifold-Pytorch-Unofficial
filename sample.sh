#!/bin/bash



python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py  \
        --num_workers 8 --teacher-model cait_XXS24_224 \
        --models deit_tiny_patch16_224 \
        --data-path ~/dataset/imagenet \
        --output_dir exp/test-fine
