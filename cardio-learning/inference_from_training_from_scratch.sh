#!/bin/bash

# Evaluation of randomly initialised (trained from scratch) models on the Cardio-Learning test set

# We cannot disclose the CardioLearning dataset because it contains Ribeiro. Once accessed Ribeiro, we have detailed how to prepare CardioLearning in the paper.

### SMALL MODEL SIZE ###

hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_random_34k_general.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_small --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_26k_cardiolearning_random_1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_small_1 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_32k_cardiolearning_random_2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_small_2 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_small_28.5k_cardiolearning_random_3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_small_3 --tta --tta_aggregation=max --n_augs=3

### BASE MODEL SIZE ###

hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_random_33k_general.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_base --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_58k_cardiolearning_random_1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_base_1 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_28k_cardiolearning_random_2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_base_2 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_base_49.5k_cardiolearning_random_3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_base_3 --tta --tta_aggregation=max --n_augs=3

### LARGE MODEL SIZE ###

hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_random_27.5k_general.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_large --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_12.5k_cardiolearning_random_1.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_large_1 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_13.5k_cardiolearning_random_2.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_large_2 --tta --tta_aggregation=max --n_augs=3
hubert-ecg-evaluate /path/to/cardiolearning_test.csv . 64 /path/to/hubert_large_56k_cardiolearning_random_3.pt --downsampling_factor=5 --label_start_index=3 --save_id=cardiolearning_random_large_3 --tta --tta_aggregation=max --n_augs=3
