#!/bin/bash

# Commands to run to perform the fine-tuning of HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All dataset as done in the paper
# Standard train/val/test splits are used
# Altoigh previous commands actually reproduce the results we obtain on PTB-XL All, we consider this dataset only as an example
# More useful information available with:   python finetune.py --help

python finetune.py 3 /path/to/ptb_all_train.csv /path/to/ptb_all_val.csv 71 8 64 auroc \
    --load_path=path/to/pretrained/model/small/hubert_ecg_small.pt \
    --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptbxl_all

python finetune.py 2 /path/to/ptb_all_train.csv /path/to/ptb_all_val.csv 71 8 64 auroc \
    --load_path=path/to/pretrained/model/base/hubert_ecg_base.pt \
    --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 \
    --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ptbxl_all

python finetune.py 3 /path/to/ptb_all_train.csv /path/to/ptb_all_val.csv 71 8 64 auroc \
    --load_path=path/to/pretrained/model/large/hubert_ecg_large.pt \
    --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 \
    --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=LARGE_ptbxl_all

