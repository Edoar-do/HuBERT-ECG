#!/bin/bash

# Fine-tuning of pre-trained HuBERT-ECG SMALL, BASE and LARGE models on Cardio-Learning (4 cross-validation folds)

# We cannot disclose the CardioLearning dataset because it contains Ribeiro. Once accessed Ribeiro, we have detailed how to prepare CardioLearning in the paper.

### SMALL MODEL SIZE ###

hubert-ecg-finetune 3 path/to/cardiolearning_train_0.csv path/to/cardiolearning_val_0.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=SMALL_cardiolearning0 --transformer_blocks_to_unfreeze=8 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_1.csv path/to/cardiolearning_val_1.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=SMALL_cardiolearning1 --transformer_blocks_to_unfreeze=8 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_2.csv path/to/cardiolearning_val_2.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=SMALL_cardiolearning2 --transformer_blocks_to_unfreeze=8 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_3.csv path/to/cardiolearning_val_3.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_small.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=SMALL_cardiolearning3 --transformer_blocks_to_unfreeze=8 --task=multi_label

### BASE MODEL SIZE ###

hubert-ecg-finetune 3 path/to/cardiolearning_train_0.csv path/to/cardiolearning_val_0.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=BASE_cardiolearning0 --transformer_blocks_to_unfreeze=12 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_1.csv path/to/cardiolearning_val_1.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=BASE_cardiolearning1 --transformer_blocks_to_unfreeze=12 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_2.csv path/to/cardiolearning_val_2.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=BASE_cardiolearning2 --transformer_blocks_to_unfreeze=12 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_3.csv path/to/cardiolearning_val_3.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_base.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=BASE_cardiolearning3 --transformer_blocks_to_unfreeze=12 --task=multi_label

### LARGE MODEL SIZE ###

hubert-ecg-finetune 3 path/to/cardiolearning_train_0.csv path/to/cardiolearning_val_0.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=LARGE_cardiolearning0 --transformer_blocks_to_unfreeze=16 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_1.csv path/to/cardiolearning_val_1.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=LARGE_cardiolearning1 --transformer_blocks_to_unfreeze=16 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_2.csv path/to/cardiolearning_val_2.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=LARGE_cardiolearning2 --transformer_blocks_to_unfreeze=16 --task=multi_label
hubert-ecg-finetune 3 path/to/cardiolearning_train_3.csv path/to/cardiolearning_val_3.csv 164 5 64 auroc --load_path=path/to/hubert_ecg_large.pt --training_steps=70000 --downsampling_factor=5 --label_start_index=3 --use_loss_weights --val_interval=5000 --finetuning_layerdrop=0.0 --random_crop --dynamic_reg --wandb_run_name=LARGE_cardiolearning3 --transformer_blocks_to_unfreeze=16 --task=multi_label
