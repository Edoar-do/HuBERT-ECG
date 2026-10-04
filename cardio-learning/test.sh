#!/bin/bash

# Evaluation of fine-tuned HuBERT-ECG models on the Cardio-Learning test set
# more information with hubert-ecg-evaluate --help

# We cannot disclose the CardioLearning dataset because it contains Ribeiro. Once accessed Ribeiro, we have detailed how to prepare CardioLearning in the paper.

### SMALL MODEL SIZE ###

hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_cardiolearning0.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_small_0 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_28k_cardiolearning1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_small_1 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_31k_cardiolearning2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_small_2 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_28.5k_cardiolearning3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_small_3 --tta --tta_aggregation=max --n_augs=3

### BASE MODEL SIZE ###

hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_30.5k_cardiolearning0.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base_0 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_25.5k_cardiolearning1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base_1 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_27.5k_cardiolearning2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base_2 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_29k_cardiolearning3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_base_3 --tta --tta_aggregation=max --n_augs=3

### LARGE MODEL SIZE ###

hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_36.5k_cardiolearning_0.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_large_0 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_52.5k_cardiolearning_1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_large_1 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_26.5k_cardiolearning_2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_large_2 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_12k_cardiolearning_3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_large_3 --tta --tta_aggregation=max --n_augs=3
