from __future__ import annotations

import argparse
import copy
import os
import random
from math import ceil

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import wandb
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from hubert_ecg import HuBERTECG, HuBERTECGConfig
from hubert_ecg.dataset import ECGDataset
from hubert_ecg.modeling import _upgrade_config

EPS = 1e-9
MINIMAL_IMPROVEMENT = 1e-3
DROPOUT_DYNAMIC_REG_FACTOR = 0.05


def dynamic_regularizer(optimizer, model, penalty):
    if penalty:
        optimizer.param_groups[0]["weight_decay"] *= 5
        for name, module in model.named_modules():
            if "dropout" in name:
                module.p += DROPOUT_DYNAMIC_REG_FACTOR
    else:
        optimizer.param_groups[0]["weight_decay"] = max(
            0.01, optimizer.param_groups[0]["weight_decay"] / 5
        )
        for name, module in model.named_modules():
            if "dropout" in name:
                module.p = max(0.1, module.p - DROPOUT_DYNAMIC_REG_FACTOR)


def _load_checkpoint(load_path: str, vocab_sizes: list, device: torch.device):
    """Load model + training state from either a legacy .pt file or a HF checkpoint dir."""
    if os.path.isdir(load_path):
        logger.info(f"Loading HF-format checkpoint from {load_path}")
        hubert = HuBERTECG.from_pretrained(load_path)
        state_path = os.path.join(load_path, "training_state.pt")
        state = torch.load(state_path, map_location="cpu")
        return hubert, state
    else:
        logger.info(f"Loading legacy .pt checkpoint from {load_path}")
        checkpoint = torch.load(load_path, map_location="cpu")
        config = _upgrade_config(
            checkpoint["model_config"],
            checkpoint.get("pretraining_vocab_sizes", vocab_sizes),
        )
        hubert = HuBERTECG(config)
        hubert.load_state_dict(checkpoint["model_state_dict"])
        state = {
            "global_step": checkpoint.get("global_step", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "best_val_accuracy": checkpoint.get("best_val_accuracy", 0.0),
            "patience_count": checkpoint.get("patience_count", 0),
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
            "lr_scheduler_state_dict": checkpoint.get("lr_scheduler_state_dict"),
            "pretraining_vocab_sizes": checkpoint.get("pretraining_vocab_sizes", vocab_sizes),
        }
        return hubert, state


def _save_checkpoint(
    hubert: HuBERTECG,
    optimizer,
    lr_scheduler,
    global_step: int,
    best_val_loss: float,
    best_val_accuracy: float,
    patience_count: int,
    vocab_sizes: list,
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
            "best_val_accuracy": best_val_accuracy,
            "patience_count": patience_count,
            "pretraining_vocab_sizes": vocab_sizes,
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "lr_scheduler_state_dict": copy.deepcopy(lr_scheduler.state_dict()),
        },
        os.path.join(ckpt_dir, "training_state.pt"),
    )


def train(args):
    device = torch.device("cuda")

    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "hubert-ecg"),
        group="self-supervised",
    )
    if args.wandb_run_name is not None:
        wandb.run.name = args.wandb_run_name

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    patience = args.patience if args.patience is not None else args.training_steps // args.val_interval
    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = max(0, 0.01 * args.weight_decay_mult)
    accumulation_steps = args.accumulation_steps
    mask_time_prob = args.mask_time_prob

    if args.largeness == "base":
        hidden_size, num_hidden_layers, num_attention_heads = 768, 12, 12
        intermediate_size, classifier_proj_size, layerdrop = 3072, 256, 0.1
    elif args.largeness == "large":
        hidden_size, num_hidden_layers, num_attention_heads = 960, 16, 12
        intermediate_size, classifier_proj_size, layerdrop = 3840, 512, 0.0
    else:  # small
        hidden_size, num_hidden_layers, num_attention_heads = 512, 8, 8
        intermediate_size, classifier_proj_size, layerdrop = 2048, 256, 0.1

    if args.resume_pretraining:
        logger.info(f"Resuming pre-training from {args.load_path}")
        hubert, state = _load_checkpoint(args.load_path, args.vocab_sizes, device)

        prev_vocab_sizes = state.get("pretraining_vocab_sizes", args.vocab_sizes)
        assert prev_vocab_sizes == args.vocab_sizes, (
            f"vocab_sizes mismatch: checkpoint has {prev_vocab_sizes}, got {args.vocab_sizes}"
        )

        global_step = state["global_step"] if args.train_iteration == state.get("train_iteration", args.train_iteration) else 0
        best_val_loss = state["best_val_loss"] if global_step > 0 else float("inf")
        best_val_accuracy = state.get("best_val_accuracy", 0.0) if global_step > 0 else 0.0
        patience_count = state["patience_count"] if global_step > 0 else 0

        if global_step == 0:
            logger.info("Switching to a new pre-training iteration — resetting label embedding and dropouts...")
            hubert.label_embedding = nn.ModuleList(
                nn.Embedding(v, hubert.config.classifier_proj_size) for v in args.vocab_sizes
            )
            for name, module in hubert.named_modules():
                if "dropout" in name and "encoder.layers" in name:
                    module.p = 0.1

        hubert.to(device)

        optimizer = optim.AdamW(
            hubert.parameters(), lr=lr, betas=betas, eps=EPS, weight_decay=weight_decay
        )
        if global_step > 0 and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            optimizer.param_groups[0]["weight_decay"] = max(0.01, optimizer.param_groups[0]["weight_decay"])
            for name, module in hubert.named_modules():
                if "dropout" in name:
                    module.p = max(0.1, module.p)

        if global_step > 0 and state.get("lr_scheduler_state_dict") is not None:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=ceil(0.08 * args.training_steps - global_step),
                num_training_steps=args.training_steps,
                last_epoch=state["lr_scheduler_state_dict"]["last_epoch"] - 1,
            )
            lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=ceil(0.08 * args.training_steps),
                num_training_steps=args.training_steps,
            )
        logger.info("Checkpoint loaded.")
    else:
        logger.info("Building model from scratch...")

        if args.downsampling_factor is None:
            conv_kernel = (10, 3, 3, 3, 3, 2, 2)
            conv_stride = (5, 2, 2, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512, 512, 512)
        elif args.downsampling_factor == 5:
            conv_kernel = (10, 3, 3, 2, 2)
            conv_stride = (4, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512)
        elif args.downsampling_factor == 10:
            conv_kernel = (10, 3, 3, 2)
            conv_stride = (4, 2, 2, 2)
            conv_dim = (512, 512, 512, 512)
        else:
            raise ValueError(f"Unsupported downsampling_factor: {args.downsampling_factor}")

        config = HuBERTECGConfig(
            ensemble_length=len(args.vocab_sizes),
            vocab_sizes=args.vocab_sizes,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            mask_time_prob=mask_time_prob,
            classifier_proj_size=classifier_proj_size,
            layerdrop=layerdrop,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            conv_dim=conv_dim,
            mask_time_length=1,
            hidden_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            activation_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            attention_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            feat_proj_dropout=max(0, DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            final_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
        )
        hubert = HuBERTECG(config)
        hubert.to(device)
        global_step = 0
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        patience_count = 0
        optimizer = optim.AdamW(
            hubert.parameters(), lr=lr, betas=betas, eps=EPS, weight_decay=weight_decay
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(0.08 * args.training_steps),
            num_training_steps=args.training_steps,
        )
        logger.info("Model built.")

    logger.info(f"Parameters: {sum(p.numel() for p in hubert.parameters()):,}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_train,
        ecg_dir_path=args.ecg_train_dir,
        downsampling_factor=args.downsampling_factor,
        features_path=args.train_features_path,
        kmeans_path=args.kmeans_path,
        memmap_path=args.memmap_train,
    )
    val_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        ecg_dir_path=args.ecg_val_dir,
        features_path=args.val_features_path,
        downsampling_factor=args.downsampling_factor,
        kmeans_path=args.kmeans_path,
        memmap_path=args.memmap_val,
    )

    assert len(args.vocab_sizes) == train_set.ensamble_length
    for v, k in zip(args.vocab_sizes, train_set.ensamble_kmeans):
        assert v == k.cluster_centers_.shape[0]

    train_dl = DataLoader(
        train_set, collate_fn=train_set.collate, num_workers=6,
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )
    val_dl = DataLoader(
        val_set, collate_fn=val_set.collate, num_workers=6,
        batch_size=args.batch_size, shuffle=False, pin_memory=True,
    )

    scaler = amp.GradScaler()
    epochs = args.training_steps // (len(train_dl) // accumulation_steps) + 1
    start_epoch = global_step // len(train_dl)

    for epoch in range(start_epoch, epochs):
        hubert.train()
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        train_losses = []

        for ecg, attention_mask, ensamble_labels in tqdm(train_dl, total=len(train_dl)):
            global_step += 1
            ecg = ecg.to(device)
            attention_mask = attention_mask.to(device)
            ensamble_labels = ensamble_labels.to(device)

            with amp.autocast():
                out = hubert(
                    ecg, attention_mask=attention_mask,
                    output_attentions=False, output_hidden_states=False, return_dict=True,
                )
                mask = out["mask_time_indices"]
                ensamble_logits = hubert.logits(out["last_hidden_state"])
                ensamble_labels_t = ensamble_labels.transpose(0, 1)

                assert len(ensamble_labels_t) == len(ensamble_logits)
                masked_loss = 0
                unmasked_loss = 0
                for labels, logits in zip(ensamble_labels_t, ensamble_logits):
                    masked_loss += F.cross_entropy(logits[mask], labels[mask])
                    unmasked_loss += F.cross_entropy(logits[~mask], labels[~mask])

                loss = (args.alpha * masked_loss + (1 - args.alpha) * unmasked_loss) / accumulation_steps

            scaler.scale(loss).backward()
            train_losses.append(loss.item())

            if global_step % accumulation_steps == 0:
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()

            if global_step % args.val_interval == 0:
                hubert.eval()
                val_losses, val_accuracies = [], []
                logger.info(f"Validating at step {global_step}...")

                for ecg, _, ensamble_labels in tqdm(val_dl, total=len(val_dl)):
                    ecg = ecg.to(device)
                    ensamble_labels = ensamble_labels.to(device).transpose(0, 1)
                    with torch.no_grad():
                        out = hubert(
                            ecg, attention_mask=None,
                            output_attentions=False, output_hidden_states=False, return_dict=True,
                        )
                        ensamble_logits = hubert.logits(out["last_hidden_state"])
                        loss = 0
                        accuracy = 0
                        for labels, logits in zip(ensamble_labels, ensamble_logits):
                            logits_t = logits.transpose(1, 2)
                            loss += F.cross_entropy(logits_t, labels)
                            accuracy += (logits_t.argmax(dim=1) == labels).float().mean()
                        accuracy /= len(ensamble_logits)
                    val_accuracies.append(accuracy.item())
                    val_losses.append(loss.item())

                val_loss = float(np.mean(val_losses))
                val_accuracy = float(np.mean(val_accuracies))
                train_loss = float(np.mean(train_losses))
                train_losses.clear()

                logger.info(f"Step {global_step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
                wandb.log({
                    f"train_loss_{args.train_iteration}": train_loss,
                    f"val_loss_{args.train_iteration}": val_loss,
                    "val_accuracy": val_accuracy,
                })

                hubert.train()

                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT:
                    best_val_loss = val_loss
                    best_val_accuracy = max(val_accuracy, best_val_accuracy)
                    patience_count = 0
                    checkpoint_name = f"hubert_{args.train_iteration}_step{global_step}_{wandb.run.id}"
                    _save_checkpoint(
                        hubert, optimizer, lr_scheduler,
                        global_step, best_val_loss, best_val_accuracy, patience_count,
                        args.vocab_sizes, args.output_dir, checkpoint_name, args.safe_serialization,
                    )
                    logger.info(f"New best val_loss={best_val_loss:.4f} — saved to {checkpoint_name}")
                    if args.dynamic_reg:
                        dynamic_regularizer(optimizer, hubert, penalty=False)

                elif val_accuracy >= best_val_accuracy + MINIMAL_IMPROVEMENT:
                    best_val_accuracy = val_accuracy
                    checkpoint_name = f"hubert_{args.train_iteration}_step{global_step}_{wandb.run.id}"
                    _save_checkpoint(
                        hubert, optimizer, lr_scheduler,
                        global_step, best_val_loss, best_val_accuracy, patience_count,
                        args.vocab_sizes, args.output_dir, checkpoint_name, args.safe_serialization,
                    )
                    logger.info(f"Val accuracy improved to {best_val_accuracy:.4f} — saved to {checkpoint_name}")
                    if args.dynamic_reg:
                        dynamic_regularizer(optimizer, hubert, penalty=False)

                else:
                    patience_count += 1
                    if args.dynamic_reg and patience_count % (patience // args.intervals_for_penalty) == 0 and patience_count != patience:
                        dynamic_regularizer(optimizer, hubert, penalty=True)
                    if patience_count == patience:
                        logger.warning(f"Early stopping at step {global_step}.")
                        wandb.log({"patience_count": patience_count})
                        return

    logger.info(f"Training done. global_step={global_step}, best_val_loss={best_val_loss:.4f}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Self-supervised pre-training of HuBERT-ECG.")

    parser.add_argument("train_iteration", type=int, choices=[1, 2, 3])
    parser.add_argument("path_to_dataset_csv_train", type=str)
    parser.add_argument("path_to_dataset_csv_val", type=str)
    parser.add_argument("val_interval", type=int)
    parser.add_argument("mask_time_prob", type=float)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("largeness", type=str, choices=["small", "base", "large"])
    parser.add_argument("alpha", type=float)
    parser.add_argument("kmeans_path", type=str)
    parser.add_argument("train_features_path", type=str)
    parser.add_argument("val_features_path", type=str)
    parser.add_argument("vocab_sizes", type=int, nargs="+")

    parser.add_argument("--ecg_train_dir", required=True, type=str,
                        help="Directory containing training ECG .npy files.")
    parser.add_argument("--ecg_val_dir", required=True, type=str,
                        help="Directory containing validation ECG .npy files.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/self-supervised",
                        help="Directory to save checkpoints.")
    parser.add_argument("--memmap_train", type=str, default=None,
                        help="Path to training memmap file (alternative to per-file .npy).")
    parser.add_argument("--memmap_val", type=str, default=None,
                        help="Path to validation memmap file.")
    parser.add_argument("--safe_serialization", action="store_true",
                        help="Save model as model.safetensors instead of pytorch_model.bin.")
    parser.add_argument("--training_steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--intervals_for_penalty", type=int, default=4)
    parser.add_argument("--resume_pretraining", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--downsampling_factor", type=int)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--dynamic_reg", action="store_true")
    parser.add_argument("--weight_decay_mult", type=int, default=1)
    parser.add_argument("--model_dropout_mult", type=int, default=0)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available.")
        raise SystemExit(1)
    if args.epochs is None and args.training_steps is None:
        raise ValueError("Provide --training_steps or --epochs.")
    if args.epochs is not None and args.training_steps is not None:
        raise ValueError("Provide either --training_steps or --epochs, not both.")
    if args.training_steps is not None and args.training_steps % args.val_interval != 0:
        raise ValueError("training_steps must be divisible by val_interval.")
    if not (0.0 < args.mask_time_prob < 1.0):
        raise ValueError("mask_time_prob must be in (0, 1).")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if args.resume_pretraining and args.load_path is None:
        raise ValueError("--load_path required with --resume_pretraining.")

    train(args)


if __name__ == "__main__":
    main()
