from __future__ import annotations

import argparse
import copy
import os
import random
from math import ceil

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.optim as optim
import wandb
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torchmetrics.classification import (
    MultilabelF1Score as F1_score,
    MultilabelRecall as Recall,
    MultilabelPrecision as Precision,
    MultilabelSpecificity as Specificity,
    MultilabelAUROC,
    MulticlassAccuracy as Accuracy,
    MulticlassAUROC,
)
from torcheval.metrics import MultilabelAUPRC as AUPRC

from hubert_ecg import HuBERTECG, HuBERTECGConfig, HuBERTECGForClassification, HuBERTECGClassificationConfig
from hubert_ecg.dataset import ECGDataset
from hubert_ecg.modeling import _upgrade_config, _remap_pos_conv_keys

EPS = 1e-9
MINIMAL_IMPROVEMENT = 1e-3
DROPOUT_DYNAMIC_REG_FACTOR = 0.05


def dynamic_regularizer(optimizer, model, penalty):
    if penalty:
        optimizer.param_groups[0]["weight_decay"] *= 5
        for name, module in model.named_modules():
            if "dropout" in name:
                module.p += 0.05
    else:
        optimizer.param_groups[0]["weight_decay"] = max(
            0.01, optimizer.param_groups[0]["weight_decay"] / 5
        )
        for name, module in model.named_modules():
            if "dropout" in name:
                module.p = max(0.1, module.p - DROPOUT_DYNAMIC_REG_FACTOR)


def _build_cls_config(backbone_config: HuBERTECGConfig, args) -> HuBERTECGClassificationConfig:
    d = {k: v for k, v in backbone_config.to_dict().items() if k != "model_type"}
    return HuBERTECGClassificationConfig(
        num_labels=args.vocab_size,
        classifier_hidden_size=args.classifier_hidden_size,
        use_label_embedding=args.use_label_embedding,
        task_type=args.task,
        **d,
    )


def _load_backbone_legacy(pt_path: str, args) -> HuBERTECGForClassification:
    """Load a pretrained backbone .pt and wrap in a new classification head."""
    checkpoint = torch.load(pt_path, map_location="cpu")
    backbone_config = _upgrade_config(
        checkpoint["model_config"],
        checkpoint.get("pretraining_vocab_sizes", [args.vocab_size]),
    )
    backbone_config.layerdrop = args.finetuning_layerdrop

    cls_config = _build_cls_config(backbone_config, args)
    hubert = HuBERTECGForClassification(cls_config)

    # Restore dropout to desired level
    state_dict = _remap_pos_conv_keys(checkpoint["model_state_dict"])
    for name, module in hubert.hubert_ecg.named_modules():
        if "dropout" in name:
            module.p = 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult

    # Prefix backbone keys so they map into hubert.hubert_ecg
    prefixed = {"hubert_ecg." + k: v for k, v in state_dict.items()}
    missing, unexpected = hubert.load_state_dict(prefixed, strict=False)
    logger.info(f"Backbone loaded. Missing keys (classifier head): {len(missing)}, unexpected: {len(unexpected)}")
    return hubert


def _load_finetuned_legacy(pt_path: str, args):
    """Resume from a legacy finetuned .pt checkpoint."""
    checkpoint = torch.load(pt_path, map_location="cpu")
    backbone_config = _upgrade_config(
        checkpoint["model_config"],
        checkpoint.get("pretraining_vocab_sizes", [args.vocab_size]),
    )
    cls_config = _build_cls_config(backbone_config, args)
    hubert = HuBERTECGForClassification(cls_config)
    state_dict = _remap_pos_conv_keys(checkpoint["model_state_dict"])
    hubert.load_state_dict(state_dict, strict=False)

    state = {
        "global_step": checkpoint.get("global_step", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "patience_count": checkpoint.get("patience_count", 0),
        "best_val_target_score": checkpoint.get(f"target_val_{args.target_metric}", 0.0),
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
        "lr_scheduler_state_dict": checkpoint.get("lr_scheduler_state_dict"),
    }
    return hubert, state


def _save_checkpoint(
    hubert: HuBERTECGForClassification,
    optimizer,
    lr_scheduler,
    global_step: int,
    best_val_loss: float,
    best_val_target_score: float,
    patience_count: int,
    target_metric: str,
    output_dir: str,
    checkpoint_name: str,
    safe_serialization: bool,
):
    ckpt_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    hubert.save_pretrained(ckpt_dir, safe_serialization=safe_serialization)
    torch.save(
        {
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "patience_count": patience_count,
            f"target_val_{target_metric}": best_val_target_score,
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "lr_scheduler_state_dict": copy.deepcopy(lr_scheduler.state_dict()),
        },
        os.path.join(ckpt_dir, "training_state.pt"),
    )


def finetune(args):
    device = torch.device("cuda")

    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "hubert-ecg"),
        group="supervised",
    )
    if args.wandb_run_name is not None:
        wandb.run.name = args.wandb_run_name

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    train_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_train,
        ecg_dir_path=args.ecg_train_dir,
        label_start_index=args.label_start_index,
        downsampling_factor=args.downsampling_factor,
        pretrain=False,
        random_crop=args.random_crop,
        memmap_path=args.memmap_train,
    )
    val_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        ecg_dir_path=args.ecg_val_dir,
        label_start_index=args.label_start_index,
        downsampling_factor=args.downsampling_factor,
        pretrain=False,
        random_crop=args.random_crop,
        memmap_path=args.memmap_val,
    )

    train_pos_weights = train_set.weights.to(device) if args.use_loss_weights else None
    val_pos_weights = val_set.weights.to(device) if args.use_loss_weights else None

    train_dl = DataLoader(
        train_set, collate_fn=train_set.collate, num_workers=6,
        batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_set, collate_fn=val_set.collate, num_workers=6,
        batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True,
    )

    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = max(0, 0.01 * args.weight_decay_mult)
    accumulation_steps = args.accumulation_steps

    task2criteria = {
        "multi_class": (
            torch.nn.CrossEntropyLoss(weight=train_pos_weights),
            torch.nn.CrossEntropyLoss(weight=val_pos_weights),
        ),
        "multi_label": (
            torch.nn.BCEWithLogitsLoss(pos_weight=train_pos_weights),
            torch.nn.BCEWithLogitsLoss(pos_weight=val_pos_weights),
        ),
        "regression": (torch.nn.MSELoss(), torch.nn.MSELoss()),
    }
    criterion_train, criterion_val = task2criteria[args.task]
    criterion_train = criterion_train.to(device)
    criterion_val = criterion_val.to(device)

    args.training_steps = args.training_steps if args.training_steps is not None else (
        (args.epochs - 1) * (len(train_dl) // accumulation_steps)
    )
    args.val_interval = len(train_dl) if args.val_interval is None else args.val_interval

    os.makedirs(args.output_dir, exist_ok=True)

    if args.resume_finetuning:
        logger.info(f"Resuming fine-tuning from {args.load_path}")
        if os.path.isdir(args.load_path):
            hubert = HuBERTECGForClassification.from_pretrained(args.load_path)
            state_path = os.path.join(args.load_path, "training_state.pt")
            state = torch.load(state_path, map_location="cpu")
        else:
            hubert, state = _load_finetuned_legacy(args.load_path, args)

        global_step = state["global_step"]
        best_val_loss = state["best_val_loss"]
        patience_count = state["patience_count"]
        best_val_target_score = state.get(f"target_val_{args.target_metric}", 0.0)
        hubert.to(device)

        if args.freezing_steps is not None and global_step < args.freezing_steps:
            hubert.set_transformer_blocks_trainable(n_blocks=0)
            hubert.set_feature_extractor_trainable(False)
        else:
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)

        parameters_group = []
        if args.layer_wise_lr and all(p.requires_grad for p in hubert.hubert_ecg.encoder.layers.parameters()):
            parameters_group.append({"params": hubert.hubert_ecg.feature_projection.parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[:args.transformer_blocks_to_unfreeze - 4].parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[args.transformer_blocks_to_unfreeze - 4:].parameters(), "lr": lr})
            parameters_group.append({"params": hubert.classifier.parameters(), "lr": 1e-5})
        else:
            parameters_group.append({"params": filter(lambda p: p.requires_grad, hubert.parameters()), "lr": lr})

        optimizer = optim.AdamW(parameters_group, betas=betas, eps=EPS, weight_decay=weight_decay)
        if state.get("optimizer_state_dict"):
            optimizer.load_state_dict(state["optimizer_state_dict"])
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc * (args.training_steps - global_step)),
            num_training_steps=args.training_steps,
        )
        if state.get("lr_scheduler_state_dict"):
            lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
        logger.info("Checkpoint loaded. Resuming fine-tuning.")

    elif args.random_init:
        logger.info("Creating model with random initialization...")
        if args.largeness == "base":
            hidden_size, num_hidden_layers, num_attention_heads = 768, 12, 12
            intermediate_size, classifier_proj_size = 3072, 256
        elif args.largeness == "large":
            hidden_size, num_hidden_layers, num_attention_heads = 960, 16, 12
            intermediate_size, classifier_proj_size = 3840, 512
        else:  # small
            hidden_size, num_hidden_layers, num_attention_heads = 512, 8, 8
            intermediate_size, classifier_proj_size = 2048, 256

        if args.downsampling_factor is None:
            conv_kernel, conv_stride = (10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)
            conv_dim = (512,) * 7
        elif args.downsampling_factor == 5:
            conv_kernel, conv_stride = (10, 3, 3, 2, 2), (4, 2, 2, 2, 2)
            conv_dim = (512,) * 5
        elif args.downsampling_factor == 10:
            conv_kernel, conv_stride = (10, 3, 3, 2), (4, 2, 2, 2)
            conv_dim = (512,) * 4
        else:
            raise ValueError(f"Unsupported downsampling_factor: {args.downsampling_factor}")

        backbone_config = HuBERTECGConfig(
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            mask_time_prob=0.0, classifier_proj_size=classifier_proj_size,
            layerdrop=args.finetuning_layerdrop, conv_kernel=conv_kernel,
            conv_stride=conv_stride, conv_dim=conv_dim, mask_time_length=1,
            hidden_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            activation_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            attention_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            feat_proj_dropout=max(0, DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            final_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
        )
        cls_config = _build_cls_config(backbone_config, args)
        hubert = HuBERTECGForClassification(cls_config)
        hubert.to(device)

        global_step = 0
        best_val_loss = float("inf")
        best_val_target_score = 0.0
        patience_count = 0
        optimizer = optim.AdamW(hubert.parameters(), lr=lr, betas=betas, eps=EPS, weight_decay=weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(0.08 * args.training_steps),
            num_training_steps=args.training_steps,
        )
        logger.info("Model created. Ready for fully supervised training.")

    else:
        logger.info(f"Fine-tuning from pretrained backbone at {args.load_path}")
        if os.path.isdir(args.load_path):
            backbone = HuBERTECG.from_pretrained(args.load_path)
            backbone.config.layerdrop = args.finetuning_layerdrop
            cls_config = _build_cls_config(backbone.config, args)
            hubert = HuBERTECGForClassification(cls_config)
            # Load backbone weights into the hubert_ecg sub-module
            prefixed = {"hubert_ecg." + k: v for k, v in backbone.state_dict().items()}
            hubert.load_state_dict(prefixed, strict=False)
        else:
            hubert = _load_backbone_legacy(args.load_path, args)

        hubert.to(device)

        global_step = 0
        best_val_loss = float("inf")
        best_val_target_score = 0.0
        patience_count = 0

        if args.freezing_steps is not None:
            hubert.set_transformer_blocks_trainable(n_blocks=0)
            hubert.set_feature_extractor_trainable(False)
        else:
            hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
            hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)

        parameters_group = []
        if args.layer_wise_lr and all(p.requires_grad for p in hubert.hubert_ecg.encoder.layers.parameters()):
            parameters_group.append({"params": hubert.hubert_ecg.feature_projection.parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[:args.transformer_blocks_to_unfreeze - 4].parameters(), "lr": 1e-7})
            parameters_group.append({"params": hubert.hubert_ecg.encoder.layers[args.transformer_blocks_to_unfreeze - 4:].parameters(), "lr": lr})
            parameters_group.append({"params": hubert.classifier.parameters(), "lr": 1e-5})
        else:
            parameters_group.append({"params": filter(lambda p: p.requires_grad, hubert.parameters()), "lr": lr})

        optimizer = optim.AdamW(parameters_group, betas=betas, eps=EPS, weight_decay=weight_decay)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(args.ramp_up_perc * args.training_steps),
            num_training_steps=args.training_steps,
        )
        logger.info("Checkpoint loaded. Ready for fine-tuning.")

    scaler = amp.GradScaler()
    epochs = args.training_steps // (len(train_dl) // accumulation_steps) + 1
    start_epoch = global_step // len(train_dl)

    task2metric = {
        "multi_label": {
            "f1-score": F1_score(num_labels=args.vocab_size, average=None),
            "recall": Recall(num_labels=args.vocab_size, average=None),
            "specificity": Specificity(num_labels=args.vocab_size, average=None),
            "precision": Precision(num_labels=args.vocab_size, average=None),
            "auroc": MultilabelAUROC(num_labels=args.vocab_size, average=None),
            "auprc": AUPRC(num_labels=args.vocab_size, average=None),
        },
        "multi_class": {
            "accuracy": Accuracy(num_classes=args.vocab_size),
            "auroc": MulticlassAUROC(num_classes=args.vocab_size),
        },
        "regression": {},
    }
    metrics = task2metric[args.task]
    assert args.target_metric in metrics, f"Target metric {args.target_metric} not available for {args.task}"
    for metric in metrics.values():
        metric.to(device)

    for epoch in range(start_epoch, epochs):
        if global_step >= args.training_steps:
            break

        hubert.train()
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        train_losses = []

        for ecg, attention_mask, labels in tqdm(train_dl, total=len(train_dl)):
            global_step += 1
            ecg = ecg.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.squeeze().to(device)

            with amp.autocast():
                out = hubert(
                    ecg, attention_mask=attention_mask,
                    output_attentions=False, output_hidden_states=False, return_dict=True,
                )
                logits = out.logits
                loss = criterion_train(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()
            train_losses.append(loss.item())

            if global_step % accumulation_steps == 0:
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()

            if args.freezing_steps is not None and global_step >= args.freezing_steps:
                hubert.set_transformer_blocks_trainable(n_blocks=args.transformer_blocks_to_unfreeze)
                hubert.set_feature_extractor_trainable(args.unfreeze_conv_embedder)
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, hubert.parameters()),
                    lr=lr, betas=betas, eps=EPS, weight_decay=weight_decay,
                )
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=ceil(args.ramp_up_perc * (args.training_steps - global_step)),
                    num_training_steps=args.training_steps,
                )

            if global_step % args.val_interval == 0:
                hubert.eval()
                val_losses = []
                for metric in metrics.values():
                    metric.reset()

                logger.info(f"Validating at step {global_step}...")
                for ecg, _, labels in tqdm(val_dl, total=len(val_dl)):
                    ecg = ecg.to(device)
                    labels = labels.squeeze().to(device)
                    with torch.no_grad():
                        out = hubert(ecg, attention_mask=None, return_dict=True)
                        logits = out.logits
                        val_losses.append(criterion_val(logits, labels).item())
                    labels_long = labels.long()
                    for metric in metrics.values():
                        metric.update(logits, labels_long)

                val_loss = float(np.mean(val_losses))
                train_loss = float(np.mean(train_losses))
                train_losses.clear()

                to_log = {"Training_loss": train_loss, "Validation_loss": val_loss}
                for name, metric in metrics.items():
                    score = metric.compute()
                    macro = score.mean()
                    logger.info(f"Validation {name} = {score}, macro: {macro:.4f}")
                    to_log[f"Validation_{name}"] = macro  # BUG FIX: was to_log[{f"Validation_{name}"]
                    if name == args.target_metric:
                        target_score = macro

                wandb.log(to_log)
                hubert.train()

                checkpoint_name = f"hubert_{args.train_iteration}_step{global_step}_finetuned_{wandb.run.id}"
                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT:
                    best_val_loss = val_loss
                    patience_count = 0
                    _save_checkpoint(
                        hubert, optimizer, lr_scheduler, global_step,
                        best_val_loss, best_val_target_score, patience_count,
                        args.target_metric, args.output_dir, checkpoint_name, args.safe_serialization,
                    )
                    logger.info(f"New best val_loss={best_val_loss:.4f} — saved.")
                    if args.dynamic_reg:
                        dynamic_regularizer(optimizer, hubert, penalty=False)

                elif target_score >= best_val_target_score + MINIMAL_IMPROVEMENT:
                    best_val_target_score = target_score
                    _save_checkpoint(
                        hubert, optimizer, lr_scheduler, global_step,
                        best_val_loss, best_val_target_score, patience_count,
                        args.target_metric, args.output_dir, checkpoint_name, args.safe_serialization,
                    )
                    logger.info(f"Val {args.target_metric} improved to {best_val_target_score:.4f} — saved.")
                    if args.dynamic_reg:
                        dynamic_regularizer(optimizer, hubert, penalty=False)

                else:
                    patience_count += 1
                    if args.dynamic_reg and patience_count % (args.patience // args.intervals_for_penalty) == 0 and patience_count != args.patience:
                        dynamic_regularizer(optimizer, hubert, penalty=True)
                    if patience_count == args.patience:
                        logger.warning(f"Early stopping at step {global_step}.")
                        wandb.log({"patience_count": patience_count})
                        return

            if global_step >= args.training_steps:
                break

    logger.info(f"Fine-tuning done. best_val_loss={best_val_loss:.4f}, best_{args.target_metric}={best_val_target_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of HuBERT-ECG.")

    parser.add_argument("train_iteration", type=int, choices=[1, 2, 3])
    parser.add_argument("path_to_dataset_csv_train", type=str)
    parser.add_argument("path_to_dataset_csv_val", type=str)
    parser.add_argument("vocab_size", type=int)
    parser.add_argument("patience", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("target_metric", type=str,
                        choices=["f1-score", "recall", "precision", "specificity", "auroc", "auprc", "accuracy"])

    parser.add_argument("--ecg_train_dir", required=True, type=str,
                        help="Directory containing training ECG .npy files.")
    parser.add_argument("--ecg_val_dir", required=True, type=str,
                        help="Directory containing validation ECG .npy files.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/supervised",
                        help="Directory to save checkpoints.")
    parser.add_argument("--memmap_train", type=str, default=None,
                        help="Path to training memmap file.")
    parser.add_argument("--memmap_val", type=str, default=None,
                        help="Path to validation memmap file.")
    parser.add_argument("--safe_serialization", action="store_true",
                        help="Save as model.safetensors instead of pytorch_model.bin.")
    parser.add_argument("--sweep_dir", type=str, default=".")
    parser.add_argument("--ramp_up_perc", type=float, default=0.08)
    parser.add_argument("--training_steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--downsampling_factor", type=int)
    parser.add_argument("--random_crop", action="store_true", default=False)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--label_start_index", type=int, default=3)
    parser.add_argument("--freezing_steps", type=int, default=None)
    parser.add_argument("--resume_finetuning", action="store_true", default=False)
    parser.add_argument("--unfreeze_conv_embedder", action="store_true", default=False)
    parser.add_argument("--transformer_blocks_to_unfreeze", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--layer_wise_lr", action="store_true", default=False)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--classifier_hidden_size", type=int, default=None)
    parser.add_argument("--use_label_embedding", action="store_true")
    parser.add_argument("--intervals_for_penalty", type=int, default=3)
    parser.add_argument("--dynamic_reg", action="store_true")
    parser.add_argument("--use_loss_weights", action="store_true")
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--largeness", type=str, choices=["small", "base", "large"])
    parser.add_argument("--weight_decay_mult", type=int, default=1)
    parser.add_argument("--model_dropout_mult", type=int, default=0)
    parser.add_argument("--finetuning_layerdrop", type=float, default=0.1)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--task", type=str, default="multi_label",
                        choices=["multi_class", "multi_label", "regression"])

    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available.")
        raise SystemExit(1)
    if args.training_steps is None and args.epochs is None:
        raise ValueError("Provide --training_steps or --epochs.")
    if args.training_steps is not None and args.val_interval is None:
        raise ValueError("--val_interval required when --training_steps is used.")
    if args.training_steps is not None and args.training_steps % args.val_interval != 0:
        raise ValueError("training_steps must be divisible by val_interval.")
    if not (0.0 <= args.ramp_up_perc <= 1.0):
        raise ValueError("ramp_up_perc must be in [0, 1].")
    if args.random_init and args.resume_finetuning:
        raise ValueError("--random_init and --resume_finetuning are mutually exclusive.")
    if not args.random_init and args.load_path is None:
        raise ValueError("--load_path required unless --random_init is set.")
    if args.freezing_steps is not None and args.training_steps is not None and args.freezing_steps > args.training_steps:
        raise ValueError("freezing_steps cannot exceed training_steps.")
    if args.random_init and args.largeness is None:
        raise ValueError("--largeness required with --random_init.")
    if args.dynamic_reg and args.patience < args.intervals_for_penalty:
        raise ValueError("patience must be >= intervals_for_penalty when using --dynamic_reg.")

    for arg, val in vars(args).items():
        print(f"  {arg} = {val}")

    finetune(args)


if __name__ == "__main__":
    main()
