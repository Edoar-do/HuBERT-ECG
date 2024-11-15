# HuBERT-ECG: A Self-Supervised Foundation Model for Broad and Scalable Cardiac Application

[![medrXiv](https://img.shields.io/badge/medRxiv-green)](???)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Models](https://huggingface.co/Edoardo-BS/HuBERT-ECG/)] 

## Abstract
Deep learning models have shown remarkable performance in electrocardiogram (ECG) analysis, but their success has been constrained by the limited availability and size of ECG datasets, resulting in systems that are more task specialists than versatile generalists. In this work, we introduce HuBERT-ECG, a foundation ECG model pre-trained in a self-supervised manner on a large and diverse dataset of 9.1 million 12-lead ECGs encompassing 164 cardiovascular conditions. By simply adding an output layer, HuBERT-ECG can be fine-tuned for a wide array of downstream tasks, from diagnosing diseases to predicting future cardiovascular events. Across diverse real-world scenarios, HuBERT-ECG achieves AUROCs from 84.3% in low-data settings to 99% in large-scale setups. When trained to detect 164 overlapping conditions simultaneously, our model delivers AUROCs above 90% and 95% for 140 and 94 diseases, respectively. HuBERT-ECG also predicts death events within a 2-year follow-up with an AUROC of 93.4%. We release models and code.

## News
- [11/2024] This repository has been made public!

## Model weights
All our models are accessible on Huffing Face [(https://huggingface.co/Edoardo-BS/HuBERT-ECG/)] under MIT license

## 📚 Citation
If you use our models or find our work useful, please consider citing us:
```
???
```


