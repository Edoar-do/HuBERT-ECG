#!/bin/bash

# Training of randomly initialised SMALL, BASE and LARGE models on Cardio-Learning

# We cannot disclose the CardioLearning dataset because it contains Ribeiro. Once accessed Ribeiro, we have detailed how to prepare CardioLearning in the paper.

### SMALL MODEL SIZE ###

hubert-ecg-finetune 3 /path/to/cardiolearning_train_0.csv /path/to/cardiolearning_val_0.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=small --wandb_run_name=cardiolearning_random0_small
hubert-ecg-finetune 3 /path/to/cardiolearning_train_1.csv /path/to/cardiolearning_val_1.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=small --wandb_run_name=cardiolearning_random1_small
hubert-ecg-finetune 3 /path/to/cardiolearning_train_2.csv /path/to/cardiolearning_val_2.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=small --wandb_run_name=cardiolearning_random2_small
hubert-ecg-finetune 3 /path/to/cardiolearning_train_3.csv /path/to/cardiolearning_val_3.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=small --wandb_run_name=cardiolearning_random3_small

### BASE MODEL SIZE ###

hubert-ecg-finetune 2 /path/to/cardiolearning_train_0.csv /path/to/cardiolearning_val_0.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=base --wandb_run_name=cardiolearning_random0_base
hubert-ecg-finetune 2 /path/to/cardiolearning_train_1.csv /path/to/cardiolearning_val_1.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=base --wandb_run_name=cardiolearning_random1_base
hubert-ecg-finetune 2 /path/to/cardiolearning_train_2.csv /path/to/cardiolearning_val_2.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=base --wandb_run_name=cardiolearning_random2_base
hubert-ecg-finetune 2 /path/to/cardiolearning_train_3.csv /path/to/cardiolearning_val_3.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=base --wandb_run_name=cardiolearning_random3_base

### LARGE MODEL SIZE ###

hubert-ecg-finetune 3 /path/to/cardiolearning_train_0.csv /path/to/cardiolearning_val_0.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=large --wandb_run_name=cardiolearning_random0_large
hubert-ecg-finetune 3 /path/to/cardiolearning_train_1.csv /path/to/cardiolearning_val_1.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=large --wandb_run_name=cardiolearning_random1_large
hubert-ecg-finetune 3 /path/to/cardiolearning_train_2.csv /path/to/cardiolearning_val_2.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=large --wandb_run_name=cardiolearning_random2_large
hubert-ecg-finetune 3 /path/to/cardiolearning_train_3.csv /path/to/cardiolearning_val_3.csv 164 12 64 auroc --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=500 --finetuning_layerdrop=0.0 --random_crop --random_init --dynamic_reg --largeness=large --wandb_run_name=cardiolearning_random3_large
