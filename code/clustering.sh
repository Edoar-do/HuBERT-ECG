#!/bin/bash

train_iteration=1 # could be 1, 2, ..., and indicates which pre-training iteration we are in
batch_size=100 # each file contains (93, kmeans_dim) tensors, meaning that partial_fit operates on 100 * 93 samples at a time
layer=8 # only for logging purposes, to better distinguish clustering on latent representation from `--layer`-th layer of HuBERT-ECG

# --- for clustering features during the first pre-training iteration ---
# We assume `dataset.csv` contains filenames identical to those used to store the corresponding features. What differs is the directory in which those files are stored.
# The features to load and cluster are stored in /path/to/directory/containing/features/to/cluster/
# In the first pre-training iteration, the features are MFCCs, naive statistical features, or a mix of both and have been extracted using the `dumping.py` script.
# Each type of features is stored in a different directory.
# The fitted kmeans models (with K ranging from `n_clusters_start` to `n_clusters_end`) will be saved in the current directory.

python path/to/cluster.py \
    /path/to/dataset.csv \
    /path/to/directory/containing/features/to/cluster/ \
    $train_iteration \
    $batch_size \
    --cluster \
    --n_clusters_start=100 \
    --n_clusters_end=300 \
    --n_steps=100 \

# --- for clustering latent representations during subsequent pre-training iterations ---
# The `dataset.csv` is likely to be the same. What changes is the directory in which the features to cluster are stored.
# In this case, and generally when `train_iteration` > 1, the features are latent representations from the `--layer`-th layer of HuBERT-ECG.

python path/to/cluster.py \
    /path/to/dataset.csv \
    /path/to/directory/containing/features/to/cluster/ \
    $train_iteration \
    $batch_size \
    --cluster \
    --n_clusters_start=500 \
    --n_clusters_end=1000 \
    --n_steps=500 \
    --layer=$layer

# --- for evaluating a previously fitted clustering model ---
# First, we delete the `--cluster` flag to enable evaluation mode.
# Here, train_iteration serves merely for wandb logging purposes.
# The features to load and evaluate are stored in /path/to/directory/containing/features/to/use/for/evaluation/
# The fitted kmeans model to evaluate is specified with the `--model_path` argument.
# The evaluation dataset is specified as the first argument.

python path/to/cluster.py \
    /path/to/evaluation_dataset.csv \
    /path/to/directory/containing/features/to/use/for/evaluation/ \
    $train_iteration \
    $batch_size \
    --model_path=path/to/the/kmeans/model/to/evaluate/ \
