from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import (
    MultilabelF1Score as F1_score,
    MultilabelRecall as Recall,
    MultilabelPrecision as Precision,
    MultilabelSpecificity as Specificity,
    MultilabelAUROC as AUROC,
    MulticlassRecall,
    MulticlassSpecificity,
)
from torcheval.metrics import MultilabelAUPRC as AUPRC
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy, MulticlassAUPRC

from hubert_ecg import HuBERTECGForClassification
from hubert_ecg.dataset import ECGDataset
from hubert_ecg.metrics import CinC2020
from hubert_ecg.modeling import _remap_pos_conv_keys, _upgrade_config
from hubert_ecg.modeling_classification import HuBERTECGClassificationConfig


def random_crop(ecg: torch.Tensor, crop_size: int = 500) -> torch.Tensor:
    """Time-aligned random crop of a batch — used for test-time augmentation.

    in : (BS, 12 * L)
    out: (BS, 12 * crop_size)
    """
    batch_size = ecg.size(0)
    ecg = ecg.view(batch_size, 12, -1)
    new_ecg = torch.zeros(batch_size, 12, crop_size, device=ecg.device)
    for i in range(batch_size):
        start = np.random.randint(0, ecg.size(-1) - crop_size)
        new_ecg[i] = ecg[i, :, start : start + crop_size]
    return new_ecg.view(batch_size, -1)


def _load_model(model_path: str) -> HuBERTECGForClassification:
    """Load a classification model from a HF checkpoint dir or a legacy .pt file."""
    if os.path.isdir(model_path):
        logger.info(f"Loading HF-format model from {model_path}")
        return HuBERTECGForClassification.from_pretrained(model_path)

    logger.info(f"Loading legacy .pt model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    config = _upgrade_config(
        checkpoint["model_config"],
        checkpoint.get("pretraining_vocab_sizes", [100]),
    )

    keys = list(checkpoint["model_state_dict"].keys())
    num_labels = checkpoint["finetuning_vocab_size"]
    use_label_embedding = checkpoint.get("use_label_embedding", False)
    if checkpoint.get("linear", True):
        classifier_hidden_size = None
    elif use_label_embedding:
        classifier_hidden_size = None
    else:
        classifier_hidden_size = checkpoint["model_state_dict"][keys[-2]].size(-1)

    cls_config = HuBERTECGClassificationConfig(
        num_labels=num_labels,
        classifier_hidden_size=classifier_hidden_size,
        use_label_embedding=use_label_embedding,
        **{k: v for k, v in config.to_dict().items() if k != "model_type"},
    )
    model = HuBERTECGForClassification(cls_config)
    state_dict = _remap_pos_conv_keys(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate(args, model: nn.Module, metrics: dict):
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv,
        ecg_dir_path=args.ecg_dir_path,
        pretrain=False,
        downsampling_factor=args.downsampling_factor,
        label_start_index=args.label_start_index,
        return_full_length=args.tta,
        memmap_path=args.memmap_test,
    )

    dataloader = DataLoader(
        testset,
        collate_fn=testset.collate,
        num_workers=5,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    model.to(device)
    model.eval()
    for metric in metrics.values():
        metric.to(device)

    if args.challenge_metric and args.task == "multi_label":
        cinc2020_metric = CinC2020(testset.diagnoses_cols, verbose=False)
        cinc2020_metric.to(device)

    logger.info("Start testing...")
    num_labels = testset.ecg_dataframe.shape[1] - args.label_start_index

    for batch_id, (ecg, _, labels) in enumerate(tqdm(dataloader, total=len(dataloader))):
        ecg = ecg.to(device)
        labels = labels.squeeze().to(device)

        ecgs = (
            [random_crop(ecg, 2500 // (args.downsampling_factor or 1)) for _ in range(args.n_augs)]
            if args.tta
            else [ecg]
        )

        probs = []
        for aug_ecg in ecgs:
            with torch.no_grad():
                out = model(aug_ecg, attention_mask=None, return_dict=True)
                logits = out.logits
            probs.append(
                torch.sigmoid(logits) if args.task == "multi_label" else torch.softmax(logits, dim=-1)
            )

        probs = (
            torch.stack(probs).mean(dim=0)
            if args.tta_aggregation == "mean"
            else torch.stack(probs).max(dim=0).values
        )

        if args.save_probs:
            dest_path = f"probs/{os.path.basename(args.path_to_dataset_csv)[:-4]}"
            os.makedirs(dest_path, exist_ok=True)
            np.save(
                os.path.join(dest_path, f"probs_{batch_id}_bs{args.batch_size}"),
                probs.cpu().numpy(),
            )

        if args.challenge_metric:
            cinc2020_metric.update((probs > 0.5).float(), labels)

        labels = labels.long()
        for metric in metrics.values():
            metric.update(probs, labels)

    performance = pd.DataFrame(columns=testset.diagnoses_cols, index=list(metrics.keys()))

    for name, metric in metrics.items():
        score = metric.compute()
        print(f"{name} = {score}, macro: {score.mean()}")
        performance.loc[name] = (
            score.cpu().numpy() if args.task == "multi_label" else score.mean().cpu().numpy()
        )

    if args.challenge_metric and args.task == "multi_label":
        score = cinc2020_metric.compute()
        print(f"CinC2020 = {score}")
        with open("cinc2020.txt", "a") as f:
            f.write(f"{args.save_id} : {score}\n")

    os.makedirs("performance", exist_ok=True)
    if args.save_id is not None:
        suffix = "_max" if args.tta_aggregation == "max" else ""
        out_path = f"./performance/performance_{args.save_id}{suffix}.csv"
        performance.to_csv(out_path)
        logger.info(f"Performance saved to {out_path}")

    logger.info("End of testing.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned HuBERT-ECG model.")

    parser.add_argument("path_to_dataset_csv", type=str, help="Path to dataset CSV.")
    parser.add_argument("ecg_dir_path", type=str, help="Directory containing ECG .npy files.")
    parser.add_argument("batch_size", type=int, help="Batch size.")
    parser.add_argument("model_path", type=str,
                        help="Path to a HF checkpoint directory or a legacy .pt file.")

    parser.add_argument("--downsampling_factor", type=int, default=None)
    parser.add_argument("--label_start_index", type=int, default=3)
    parser.add_argument("--save_id", type=str, default=None)
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--tta_aggregation", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--n_augs", type=int, default=3)
    parser.add_argument("--challenge_metric", action="store_true", default=False)
    parser.add_argument("--task", type=str, default="multi_label",
                        choices=["multi_label", "multi_class", "regression"])
    parser.add_argument("--save_probs", action="store_true", default=False)
    parser.add_argument("--memmap_test", type=str, default=None,
                        help="Path to test memmap file (alternative to per-file .npy).")

    args = parser.parse_args()

    model = _load_model(args.model_path)
    num_labels = model.config.num_labels

    task2metric = {
        "multi_label": {
            "test_f1_score": F1_score(num_labels=num_labels, average=None),
            "test_recall": Recall(num_labels=num_labels, average=None),
            "test_precision": Precision(num_labels=num_labels, average=None),
            "test_specificity": Specificity(num_labels=num_labels, average=None),
            "test_auroc": AUROC(num_labels=num_labels, average=None),
            "test_auprc": AUPRC(num_labels=num_labels, average=None),
        },
        "multi_class": {
            "test_accuracy": MulticlassAccuracy(num_classes=num_labels),
            "test_auroc": MulticlassAUROC(num_classes=num_labels),
            "test_recall": MulticlassRecall(num_classes=num_labels),
            "test_specificity": MulticlassSpecificity(num_classes=num_labels),
            "test_auprc": MulticlassAUPRC(num_classes=num_labels),
        },
        "regression": {},
    }

    evaluate(args, model, task2metric[args.task])


if __name__ == "__main__":
    main()
