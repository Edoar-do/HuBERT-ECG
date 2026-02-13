# HuBERT-ECG as a Self-Supervised Foundation Model for Broad and Scalable Cardiac Application

[![medrXiv](https://img.shields.io/badge/medRxiv-green)](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v2)
License: CC BY-NC 4.0


📢 [[Models](https://huggingface.co/Edoardo-BS)] 

## Abstract
Deep learning models have shown remarkable performance in electrocardiogram (ECG) analysis, but the limited availability and size of ECG datasets have constrained their success, resulting in systems that are more task specialists than versatile generalists. To counter this, we introduce HuBERT-ECG, a novel self-supervised foundation ECG model pre-trained on a large and diverse dataset of 9.1 million 12-lead ECGs encompassing 164 cardiovascular conditions. By simply adding a proper output layer, HuBERT-ECG can be fine-tuned for a wide array of downstream tasks, from diagnosing diseases to predicting future cardiovascular events. Across diverse real-world scenarios, HuBERT-ECG achieves AUROCs from 0.843 on small datasets to 0.99 on larger sources. When fine-tuned to detect 164 overlapping conditions simultaneously, our model delivers AUROCs above 0.9 and 0.95 for up to 140 and 97 diseases, respectively. HuBERT-ECG can also predict death events within a 2-year follow-up with AUROCs up to 0.91. We release pre-trained models and code as building baselines.

## News
- [06/2025] A new medrxiv version has been updated with new results, findings and insights!
- [12/2024] Reproducibility has never been easier! Training, validation, and test splits ready to use in the reproducibility folder!
- [12/2024] Pre-trained models are easily downloadable from Hugging Face using `AutoModel.from_pretrained`
- [11/2024] Pre-trained models are freely available on HuggingFace
- [11/2024] This repository has been made public!

## Model weights
All our models are accessible on Hugging Face [(https://huggingface.co/Edoardo-BS)] under CC BY-NC 4.0 license

## ⚠️ How to use HuBERT-ECG on your own datasets ⚙️
### Create your dataset
First, you need to take all your 12-lead ECGs and store them into a directory at the following path `ecg_dir_path` with `.npy` extension. Before saving them, we recommend to preprocess them using the preprocessing function in `utils.py` and sample them at multiples of 100 Hz so that downsampling to 100 Hz (see `__get_item__()` in `dataset.py`) can be easily accomplished by specifying the `downsampling_factor` when calling training scripts.
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
--downsampling_factor=5 \ # downsampling factor to feed the model with ECGs sampled at 100 Hz (this assumes you saved them at 500 Hz but can be any multiple of 100 Hz)
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



## Installation
Clone this repository and install all the necessary dependecies written in the `requirements.txt` file with ```pip install -r requirements.txt```.
Full installation time may take up to 1 minute.

## Reproducibility
In the `reproducibility` folder you can find all train, validation, and test splits we used in our work as .csv files. You simply have to follow the instructions in the `reproducibility/README.md` to reproduce our results.
In the `finetune.sh`, there is ready-to-launch code for reproduce fine-tuning of pre-trained models while in the `test.sh` scfipt there's the code for evaluation of fine-tuned models.
Similarly, `train_from_scratch` allows you to replicate every training from scratch, that is, train in a fully supervised manner the same models with random initialization. `inference_from_training_from_scratch.sh` contains the code to run evaluation of these trained-from-scratch models.
The forward pass on a single instance takes less than 1 second on an A100 GPU node, which is also the machine we ran our experiments and evaluations on.
Experiments on `Google Colab` show that even the LARGE model size can easily fit into a T4 GPU.
The splits were used in cross-validation experiments/evaluations to also mitigate the performance difference that can be be observed when using different hardware and machiens.

## 📚 Citation
If you use our models or find our work useful, please consider citing us:
```
https://doi.org/10.1101/2024.11.14.24317328
```


