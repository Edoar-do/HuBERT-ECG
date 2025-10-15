#!/bin/bash

train_iteration=1 # could be 1, 2, ...
path_to_dataset_train=/path/to/your/training/dataset.csv
path_to_dataset_val=/path/to/your/validation/dataset.csv
val_interval=4000
mask_time_prob=0.33 # optimal masking ratio found empirically (see supplementary material)
batch_size=448
alpha=1.0
kmeans_path=/path/to/the/file/containing/paths/to/kmeans/models.txt # this file can reference multiple pkl files containins multiple kmeans models in case of multi-task pre-training
train_features_path=/path/to/the/dir/containing/the/training/clustered/features/for/this/iteration/
val_features_path=/path/to/the/dir/containing/the/validation/clustered/features/for/this/iteration/
vocab_sizes="100" # this space-separated list of integers must match the number of kmeans models used and their respective `K`s
training_steps=80000 # in subsequent pre-training itearations, 800k steps were performed for the BASE model and 400k steps for SMALL and LARGE
largeness="base"

python /path/to/pretrain.py \
  $train_iteration \
  $path_to_dataset_csv_train \
  $path_to_dataset_csv_val \
  $val_interval \
  $mask_time_prob \
  $batch_size \
  $largeness \
  $alpha \
  $kmeans_path \
  $train_features_path \
  $val_features_path \
  $vocab_sizes \
  --patience=10 \
  --training_steps=$training_steps \
  --intervals_for_penalty=4 \
  --downsampling_factor=5 \
  --dynamic_reg \
  
  

  
