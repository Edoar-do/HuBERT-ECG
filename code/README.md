# For future users and fine-tuners

Before start reading the contents of this folder, please be aware that you need to pre-process all your ECGs as this is not done on-the-fly and HuBERT-ECG was trained on bandpass-filtered and rescaled signals.
While the `ECGDataset` can be tasked with downsampling (see the `downsampling_factor` parameter) and cropping to 5 seconds, filtering and rescaling must be performed a-priori. The .csv file will therefore reference preprocessed files ready to be loaded and used. The preprocessing functions can be found in the `utils.py` file.

# Code explanation

## Dumping
`dumping.py` contains the code and entry points to compute and dump feature descriptors of raw ECG fragments. These descriptors include:
- time-frequency feautures
- 39 MFCC coefficients
- time-frequency features + 13 MFCC coefficients
- latent representations extracted from $i^{th}$ encoding layer, $i = 0, 1, 2..., 11$

## Clustering
After dumping ECG feature descriptors, one can proceed with the offline clustering step, that is, clustering the feature descriptor and fit a K-means clustering model. 
`clusteri.py` implements such a step, saves the resulting model, which is necessary to produce labels to use in the pre-training, and provides evaluation functions to quantify the clustering quality. 
The `clustering.sh` script help understand how to start this operation.

## Dataset
The `dataset.py` file contains the ECGDataset implementation, responsible of iterating over a csv file representing an ECG dataset (normally train/val/test sets) and provinding the data loader with ECGs, ECG feature descriptors, and ECG up/downstream labels.

## HuBERT-ECG
The architecture of HuBERT-ECG one sees during pre-training is provided in the `hubert_ecg.py` file, while the archicture one sees during fine-tuning or training from scratch is provided in the `hubert_ecg_classification.py` file.
The difference consists in projection & look-up embedding matrices present in the former architecture that are replaced by the classification head present in the latter one.

## Pre-training
`pretrain.py` contains the code to pre-train HuBERT-ECG in a self-supervised manner. `python pretrain.py --help` is highly suggested. In addition, `pretraining.sh` is also helpful.

## Fine-tuning
`finetune.py` contains the code to fine-tune and train from scratch HuBERT-ECG in a supervised manner. `python finetune.py --help` is highly suggested as well as a look at `finetune.sh`

## Testing/Evaluation
`test.py` contains the code to evaluate fine-tuned or fully trained HuBERT-ECG instances on test data. `python test.py --help` is highly suggested as well as a look at `test.sh`

## Utils
`utils.py` contains utility functions, including those for preprocessing.

## How to use HuBERT-ECG on your own datasets ⚙️
### Create your dataset
First, you need to take all your 12-lead ECGs and store them into a directory at the following path `ecg_dir_path` with `.npy` extension. Before saving them, we recommend to preprocess them using the preprocessing function in `utils.py` and resample them at 500 Hz so that downsampling to 100 Hz and random cropping (see `__get_item__()` in `dataset.py`) can be easily accomplished by specifying the `downsampling_factor = 5` when calling training scripts.
Second, create a `.csv` file with the following columns: `filename`, opt. `age`, opt. `sex`, `label1`, ..., `labelN`. The `label` columns represent the classes/labels HuBERT-ECG has to learn and are filled in a multi-hot fashion for multi-label classification problems. For multi-class classification, binary classification and regression tasks, there should be only one `label` column, containing integer class indices from `0` to `C-1` or real values to predict in case of regression tasks. NOTE: binary classification is treated as a 2-class problem. the `filename` column is used in conjuction with `ecg_dir_path` to reference you ECG files but can optionally contain the entire path to those files, not only their basename. At the end of this process, for example, you should have something like this in case of multi-label classification
```
filename,age,sex,Atrial Fibrillation,Sinus Bradycardia,Normal,...
ecg_0.npy,65,male,0,0,1,...
ecg_1.npy,38,female,1,0,0,...
```
or like this (y ∈ [0, C-1] or y ∈ R) in case of multi-class classification or regression
```
filename,age,sex,multi_class_label_or_regression_target
ecg_0.npy,65,male,y
ecg_1.npy,38,female,y
```
You can then use traditional sklearn packages and function to split this dataset into training, validation, and test splits. You can even add more columns if you need them but **the important thing is that `label` columns are always the last ones**

### Start fine-tuning
After downloading model checkpoint from Hugging-Face, perhaps in `.pt` format, you can call the `finetune.py` script this way:
```
python finetune.py \
3 \ # train iteration --> just leave 3
path/to/your_dataset_train.csv \ # your training set in .csv format with the above structure
path/to/your_dataset_val.csv \ # your validation set in .csv format with the above structure
6 \ # num_classes/labels --> should match the number of label columns in multi-label classification columns or [C] in case of multi-class tasks; should be 1 for regression
5 \ # patience for early stopping
64 \ # batch size 
auroc \ # target metric to monitor for checkpointing
--ecg_dir_path=/path/to/your/ECG/files # optional if the csv file references ECG by full path
--load_path=path/to/hubert_ecg_small.pt \ # path to the m.pt model you have downloaded from hugging-face
--training_steps=70000 \ # number of training steps to perform
--downsampling_factor=5 \ # downsampling factor to feed the model with ECGs sampled at 100 Hz (this assumes you saved them at 500 Hz)
--label_start_index=3 \ # the index of the csv file at which you start with label column
--use_loss_weights \ # whether to use weights in the loss function computation
--transformer_blocks_to_unfreeze=8 \ # number of transformer blocks/layers to finetune from the last one backwwards. up to 8 for small size, 12 for base size, 16 for large size
--val_interval=5000 \ # how many steps to wait before validating
--finetuning_layerdrop=0.0 \ # layerdrop for regularization
--wandb_run_name=your_wandb_run_name
```
The `finetune.py` has many other interesting parameters to explore. Take a look at `python finetune.py --help`. Normally, finetuning is very easy and simply requires sweeping over layerdrop values in case of overfitting, meaning that things like layer-wise learning rate scheduling, extensive hyper-parameter tunings etc. are not necessary.

### Test your finetuned model
After finetuning, you will see all finetuned model checkpoints at the following path: `SUPERVISED_MODEL_CKPT_PATH = "/models/checkpoints/supervised/"`. 
The finetuning script saves a checkpoint whenever the validation loss or the validation target metric improves. We suggest to take a look at wandb metric/loss trend to choose your checkpoint for testing as there might be cases where your target metric improves even if the validation loss doesn't.

After selecting your checkpoint, just run the `test.py` script like this
```
python test.py \
/path/to/your_dataset_test.csv \
/path/to/your/ECG/files/ \ 
64 \ # batch size # not really important since we accumulate but helps with speed
/path/to/finetuned/checkpoint.pt \
--downsampling_factor=5 \
--save_id=id_of_performance_summary_in_csv \
--label_start_index=3 # where labels start in the csv column list
```
After testing, you can analyse performance at `f"./performance/performance_{args.save_id}.csv"`.
If your ECG last at least 10 second, you can test with `--tta` enabled (test-time augmentation) and select the number of random crops/augmentated views to use to compute the final prediction (`--n_augs`). You can even select how to aggreagte predicitons ( `--tta_aggregation` either `mean` or `max`. You can optionally save probabilities, perhaps to compute confidence intervals via bootstrapping (see `utils.py`), using `--save_probs`)

