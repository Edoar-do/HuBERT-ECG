# ECG-pretraining
This is a Deep Learning project that aims to pretrain a model in a self-supervised fashion and robustly. The final goal is to employ this single model in a wide variety of downstream tasks by conducting a preliminar simple finetuning.

## Data
The data used in this project comprises 12-lead ECGs collected from multiple sources, both public and private, around the world. The dataset include more than 8.3 million ECGs collected from both patients in sinus rhythm and patients affected by more than 140 different conditions (sometimes even co-occuring). The list of countries that contributed to the creation of this dataset include:
- China
- Brasil
- Germany
- Russia
  
The availability of large and diverse datasets is crucial in order to train models that are robust to the varying conditions of the real-world contexts.

## Methods
Although millions of these ECGs are labelled, the majority of them are not. To build a robust and generalizable model this circumstance suggests a self-supervised training approach where there is no need of labels. In fact, supervised approaches have proved useful to build task-specific models whose predictive capabilities are hardly transferable on different tasks without drops in performance, unless a heavy fine-tuning is performed.  Given this context, a lot of self-supervised approaches have been developed recently, but the most promising was introduced by Devlin et al. in [BERT](https://arxiv.org/abs/1810.04805): the so called Masked Language Modelling. This procedure basically masks a random portion of input tokens and asks the model to learn what were the original tokens given the unmasked ones in the sequence. Nonetheless, BERT was not created to process waveforms but sentences, hence a different approach is required and an immediate solution is [HuBERT](https://arxiv.org/abs/2106.07447). HuBERT was designed to process audio signals and was pretrained in predicting masked hidden units, which are pseudo-labels provided by a clustering (teacher) model fitted on features extracted from shards of the input signal and, in subsequent training iterations, from mid-layers of the Transformer's encoder.

