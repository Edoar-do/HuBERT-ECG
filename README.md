# HuBERT-ECG as a Self-Supervised Foundation Model for Broad and Scalable Cardiac Application

[![medrXiv](https://img.shields.io/badge/medRxiv-green)](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328)
License: CC BY-NC 4.0


📢 [Models(https://huggingface.co/Edoardo-BS)] on Hugging Face 🤗

## Abstract
The electrocardiogram (ECG) is a widely accessible tool for cardiovascular assessment, and thegrowing availability of ECG datasets has enabled the emergence of ECG foundation models. However, such foundation models often lack extensive evaluation across clinically heterogeneous downstream tasks extending beyond conventional rhythm and conduction analysis. We present HuBERT-ECG, a self-supervised foundation ECG model pre-trained on 9.1 million 12-lead ECGs from four countries and diverse patient populations, and evaluated through fine-tuning on 21 independent datasets spanning more than 1.6k diagnostic and prognostic targets across adults and paediatric cohorts, including single-lead settings. These tasks cover conditions for which the ECG is the primary diagnostic modality, provides supportive but non-definitive diagnostic information, or enables acute-care prediction and prognostic modelling. Available in three model sizes to characterise scaling behaviour and support diverse computational constraints, HuBERT-ECG achieves AUROC ranging from 84% to 99% on ECG-primary diagnostic tasks, 76% to 97% on supportive diagnostic tasks, 74% to 91% on prognostic prediction tasks, and 88% to 92% on single-lead ECG benchmarks. Moreover, a large-scale multitask fine-tuning across 2.4 million subjects and 164 tasks simultaneously shows that AUROC further increases for clinically relevant tasks without extra task-specific supervision. We release pretrained models and code as building baselines.

## News
- [06/2026] Cardio-Learning model checkpoints available on Hugging Face 🤗 for a more disease-oriented baseline to further finetune!
- [06/2026] A **substantial** new version on medrxiv!
- [06/2025] A new version on medrxiv with new results, findings and insights!
- [12/2024] Reproducibility has never been easier! Training, validation, and test splits ready to use in the reproducibility folder!
- [12/2024] Pre-trained models are easily downloadable from Hugging Face 🤗 using `AutoModel` APIs!
- [11/2024] Pre-trained models are freely available on HuggingFace
- [11/2024] This repository has been made public!

## Model weights
All our models are accessible on Hugging Face 🤗 [(https://huggingface.co/Edoardo-BS)] under CC BY-NC 4.0 license

## Installation

### Option A: Python virtual environment (recommended for all users)

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
git clone https://github.com/Edoardo-BS/HuBERT-ECG.git
cd HuBERT-ECG
pip install -e .
```

### Option B: Conda

```bash
conda create -n hubert-ecg python=3.10
conda activate hubert-ecg
git clone https://github.com/Edoardo-BS/HuBERT-ECG.git
cd HuBERT-ECG
pip install -e .
```

### Option C: Docker

Use a PyTorch base image for CUDA support:

```Dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

RUN pip install --upgrade pip

WORKDIR /hubert-ecg
COPY . .

RUN pip install -e .

CMD ["/bin/bash"]
```

Build and run:

```bash
git clone https://github.com/Edoardo-BS/HuBERT-ECG.git
cd HuBERT-ECG
docker build -t hubert-ecg .
docker run --gpus all -it hubert-ecg bash
```

---

After installation, you'll have the `hubert_ecg` Python package and four CLI entry points:
`hubert-ecg-pretrain`, `hubert-ecg-finetune`, `hubert-ecg-evaluate`, `hubert-ecg-convert`.

## Loading pre-trained models from Hugging Face

After `pip install -e .`, the `hubert_ecg` package registers `HuBERTECG` and
`HuBERTECGForClassification` with the HuggingFace AutoModel API:

```python
import hubert_ecg  # registers custom model types with AutoModel
from transformers import AutoModel

model = AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-base")
```

To load a legacy `.pt` checkpoint (single-file format from the original release):

```python
from hubert_ecg import HuBERTECG
model = HuBERTECG.from_pretrained_legacy("path/to/old_checkpoint.pt")
```

To convert a legacy checkpoint to HF format (and optionally push to Hub):

```bash
hubert-ecg-convert --input old_checkpoint.pt --output ./converted/
hubert-ecg-convert --input old_checkpoint.pt --output ./converted/ \
    --push_to_hub --hub_model_id YourOrg/hubert-ecg-base --hub_token $HF_TOKEN
```

## Training

> All scripts accept `--help` for the full argument list.
> Set `WANDB_ENTITY` and `WANDB_PROJECT` environment variables before running.

- For <b>self-supervised pre-training</b> take a look at `scripts/hubert-ecg-pretrain --help` and `scripts/pretrain.sh`.
- For <b>supervised finetuning</b> take a look at `scripts/hubert-ecg-finetune --help` and `scripts/finetune.sh`.
- For <b>test-time evaluation</b> take a look at `scripts/hubert-ecg-evaluate --help` and `scripts/test.sh`.

### Using memmap datasets (faster I/O for large datasets)

Build a memmap file once; it produces `train.memmap` (binary) and `train.memmap.index.csv` (index CSV with a `memmap_idx` column):

```python
from hubert_ecg.utils import build_memmap_dataset
index_csv = build_memmap_dataset("train.csv", "/data/ecg/train", "train.memmap")
# index_csv == "train.memmap.index.csv"
```

Then use the index CSV as the dataset CSV and pass the `.memmap` path to the training script:

```bash
hubert-ecg-finetune 1 train.memmap.index.csv val.csv ... \
    --memmap_train train.memmap \
    --ecg_train_dir /data/ecg/train ...
```

## Reproducibility
In the `reproducibility` folder you can find all train, validation, and test splits we used in our work as .csv files. You simply have to follow the instructions in the `reproducibility/README.md` to reproduce our results.
In `scripts/finetune.sh`, there is ready-to-launch code to reproduce fine-tuning of pre-trained models while `scripts/test.sh` contains the evaluation commands.
Similarly, `scripts/train_from_scratch.sh` allows you to replicate every training from scratch. `scripts/inference_from_training_from_scratch.sh` contains the code to run evaluation of these trained-from-scratch models.
The forward pass on a single instance takes less than 1 second on an A100 GPU node, which is also the machine we ran our experiments and evaluations on.

Experiments on `Google Colab` show that even the LARGE model size can easily fit into a T4 GPU.
The splits were used in cross-validation experiments/evaluations to also mitigate the performance difference that can be observed when using different hardware and machines.

## 📚 Citation
If you use our models or find our work useful, please consider citing us:
```
@article{coppola2024hubert,
  title={HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications},
  author={Coppola, Edoardo and Savardi, Mattia and Massussi, Mauro and Adamo, Marianna and Metra, Marco and Signoroni, Alberto},
  journal={medRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}
```


