# Cardio-Learning

Cardio-Learning is the large-scale multitask dataset (2.4 million subjects, 164 tasks) used in the paper for multitask fine-tuning.

Since the Ribeiro dataset, a significant part of Cardio-Learning, cannot be disclosed, the dataset itself cannot be released and the results on it are hard to reproduce. Once you have obtained access to Ribeiro, the paper details how to prepare Cardio-Learning.

## Folder contents

- `cardio-learning-labels.pkl`: the ordered label columns of the dataframe used to reference the Cardio-Learning examples. Use it to map the outputs of the `hubert_ecg_{size}_cardio_learning` models on [Hugging Face](https://huggingface.co/Edoardo-Coppola) to their labels.
- `finetune.sh`: fine-tuning of the pre-trained SMALL, BASE and LARGE models on Cardio-Learning (4 cross-validation folds).
- `test.sh`: evaluation of the fine-tuned models on the Cardio-Learning test set.
- `train_from_scratch.sh`: training of randomly initialised SMALL, BASE and LARGE models on Cardio-Learning.
- `inference_from_training_from_scratch.sh`: evaluation of the models trained from scratch.

All paths in the scripts are placeholders (`path/to/...`) to replace with your own files. Run `hubert-ecg-finetune --help` and `hubert-ecg-evaluate --help` for the full argument list.
