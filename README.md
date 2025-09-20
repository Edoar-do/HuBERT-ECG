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

**Remember to pre-process your data before feeding HuBERT-ECG. Take a look at Data and Preprocessing section in the paper**

## Installation
Clone this repository and install all the necessary dependecies written in the `requirements.txt` file with ```pip install -r requirements.txt```.
Full installation time may take up to 1 minute.

## Reproducibility
In the `reproducibility` folder you can find all train, validation, and test splits we used in our work as .csv files. You simply have to follow the instructions in the `reproducibility/README.md` to reproduce our results.
As an example, you can easily fine-tune and evaluate an instance of HuBERT-ECG on PTB-XL All dataset, as shown in .sh scripts `finetune.sh` and `test.sh`.
Thw forward pass on a single instance takes less than 1 second on an A100 GPU node, which is also the machine we ran our experiments and evaluations on.
The splits were used in cross-validation experiments/evaluations to also mitigate the performance difference that can be be observed when using different hardware and machiens.

## 📚 Citation
If you use our models or find our work useful, please consider citing us:
```
https://doi.org/10.1101/2024.11.14.24317328
```


