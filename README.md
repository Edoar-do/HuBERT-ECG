# ECG-pretraining
This is Deep Learning project that aims at building a robust pre-trained model in a self-supervised fashion with transferable capabilities when dealing with 12-lead ECGs interpretation.

## Data
The data used in this project comprises 12-lead ECGs collected from multiple sources, both public and private, around the world. The dataset include more than 8.3 million ECGs collected from both patients in sinus rhythm and patients affected by more than 130 different conditions (sometimes even co-occuring conditions). The list of countries that contributed to the creation of this dataset include:
- China
- Brasil
- Germany
- Russia

## Methods
Although millions of these ECGs are labelled, the majority of them are not. To build a robust and generalizable model this circumstance suggests a self-supervised training approach where there is no need of labels. In fact, supervised approaches have been demostrated useful to build task-specific models that are hardly transferable on different tasks, unless a heavy fine-tuning is performed.  Given this context, a lot of self-supervised approaches have been developed recently, but the most promising was introduced by Devlin et al. in [BERT](https://arxiv.org/abs/1810.04805): the so called Masked Language Modelling. This procedure basically masks a random portion of input tokens and asks the model to learn what were the original tokens given the unmasked tokens in the sequence.

