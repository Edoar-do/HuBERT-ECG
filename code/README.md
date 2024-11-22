# Code explanation

## Dumping
`dumping.py` contains the code and entry points to compute and dump feature descriptors of raw ECG fragments. These descriptors include:
- time-frequency feautures
- 39 MFCC coefficients
- time-frequency features + 13 MFCC coefficients
- latent representations extracted from $i^{th}$ encoding layer, $i = 0, 1, 2..., 11$

## Clustering
After dumping ECG feature descriptors, one can proceed with the offline clustering step, that is, clustering the feature descriptor and fit a K-means clustering model. 
`clustering.py` implements such a step, saves the resulting model, which is necessary to produce labels to use in the pre-training, and provides evaluation functions to quantify the clustering quality

## Dataset
The `dataset.py` file contains the ECGDataset implementation, responsible of iterating over a csv file representing an ECG dataset (normally train/val/test sets) and provinding the data loader with ECGs, ECG feature descriptors, and ECG up/downstream labels.

## HuBERT-ECG
The architecture of HuBERT-ECG one sees during pre-training is provided in the `hubert_ecg.py` file, while the archicture one sees during fine-tuning or training from scratch is provided in the `hubert_ecg_classification.py` file.
The difference consists in projection & look-up embedding matrices present in the former architecture that are replaced by the classification head present in the latter one.

## Pre-training
`pretrain.py` contains the code to pre-train HuBERT-ECG in a self-supervised manner. `python pretrain.py --help` is highly suggested.

## Fine-tuning
`finetune.py` contains the code to fine-tune and train from scratch HuBERT-ECG in a supervised manner. `python finetune.py --help` is highly suggested as well as a look at `finetune.sh`

## Testing/Evaluation
`test.py` contains the code to evaluate fine-tuned or fully trained HuBERT-ECG instances on test data. `python test.py --help` is highly suggested as well as a look at `test.sh`

## Utils
`utils.py` contains utility functions.
