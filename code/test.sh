#!/bin/bash

# Command to run to test HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All test set
# More useful information with:   python test.py --help
# Finetuned models are available

python test.py /path/to/ptb_all_test.csv . 64 \
    ../path/to/finetuned/model/hubert_small_12.5k_ptbAll.pt \
    --downsampling_factor=5 \
    --label_start_index=4 \
    --tta \
    --save_id=ptb_all_small \
    --tta_aggregation=max

python test.py /path/to/ptb_all_test.csv . 64 \
    ../path/to/finetuned/model/hubert_base_9k_ptbAll.pt \
    --downsampling_factor=5 \
    --label_start_index=4 \
    --tta \
    --save_id=ptb_all_base \
    --tta_aggregation=max

python test.py /path/to/ptb_all_test.csv . 64 \
    ../path/to/finetuned/model/hubert_large_8.5k_ptbAll.pt \
    --downsampling_factor=5 \
    --label_start_index=4 \
    --tta \
    --save_id=ptb_all_large \
    --tta_aggregation=max

