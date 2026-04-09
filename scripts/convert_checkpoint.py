"""
convert_checkpoint.py — Legacy checkpoint converter for HuBERT-ECG.

NOTE: This script exists purely for backward compatibility.
It converts checkpoints saved in the legacy format
(a Python dict with 'model_state_dict' and 'model_config' keys produced by
the original code/ training scripts) into the HuggingFace save_pretrained
format (config.json + pytorch_model.bin or model.safetensors).

New checkpoints produced by scripts/pretrain.py and scripts/finetune.py are
already in HF format and do NOT need conversion.

Typical usage
-------------
# Convert a pre-trained backbone:
  python -m scripts.convert_checkpoint --input old_pretrain.pt --output ./converted/

# Convert a fine-tuned classification checkpoint:
  python -m scripts.convert_checkpoint --input old_finetune.pt --output ./converted/ --classification

# Convert and push to HuggingFace Hub:
  python -m scripts.convert_checkpoint --input old_pretrain.pt --output ./converted/ \\
      --push_to_hub --hub_model_id YourOrg/hubert-ecg-base --hub_token $HF_TOKEN

# Write model.safetensors instead of pytorch_model.bin:
  python -m scripts.convert_checkpoint --input old_pretrain.pt --output ./converted/ --safe_serialization
"""

from __future__ import annotations

import argparse
import os

import torch
from loguru import logger

from hubert_ecg import HuBERTECG, HuBERTECGForClassification
from hubert_ecg.modeling import _remap_pos_conv_keys, _upgrade_config
from hubert_ecg.modeling_classification import HuBERTECGClassificationConfig


def _convert_backbone(checkpoint: dict, output_dir: str, safe_serialization: bool) -> HuBERTECG:
    config = _upgrade_config(
        checkpoint["model_config"],
        checkpoint.get("pretraining_vocab_sizes", [100]),
    )
    model = HuBERTECG(config)
    state_dict = _remap_pos_conv_keys(checkpoint["model_state_dict"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    logger.info(f"Backbone saved to {output_dir}")
    return model


def _convert_classification(checkpoint: dict, output_dir: str, safe_serialization: bool) -> HuBERTECGForClassification:
    backbone_config = _upgrade_config(
        checkpoint["model_config"],
        checkpoint.get("pretraining_vocab_sizes", [100]),
    )
    num_labels = checkpoint.get("finetuning_vocab_size", 2)
    use_label_embedding = checkpoint.get("use_label_embedding", False)
    linear = checkpoint.get("linear", True)

    if linear or use_label_embedding:
        classifier_hidden_size = None
    else:
        keys = list(checkpoint["model_state_dict"].keys())
        classifier_hidden_size = checkpoint["model_state_dict"][keys[-2]].size(-1)

    cls_config = HuBERTECGClassificationConfig(
        num_labels=num_labels,
        classifier_hidden_size=classifier_hidden_size,
        use_label_embedding=use_label_embedding,
        **{k: v for k, v in backbone_config.to_dict().items() if k != "model_type"},
    )
    model = HuBERTECGForClassification(cls_config)
    state_dict = _remap_pos_conv_keys(checkpoint["model_state_dict"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    logger.info(f"Classification model saved to {output_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a legacy HuBERT-ECG .pt checkpoint to HuggingFace save_pretrained format. "
            "Optionally push the converted model to the HuggingFace Hub."
        )
    )
    parser.add_argument("--input", required=True, type=str,
                        help="Path to the legacy .pt checkpoint file.")
    parser.add_argument("--output", required=True, type=str,
                        help="Output directory for the converted model.")
    parser.add_argument("--classification", action="store_true",
                        help="Set if the checkpoint is a fine-tuned classification model.")
    parser.add_argument("--safe_serialization", action="store_true",
                        help="Save as model.safetensors instead of pytorch_model.bin.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the converted model to the HuggingFace Hub after saving.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="HuggingFace Hub model ID (e.g. YourOrg/hubert-ecg-base). "
                             "Required when --push_to_hub is set.")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="HuggingFace write token. Falls back to $HF_TOKEN env var.")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.push_to_hub and args.hub_model_id is None:
        raise ValueError("--hub_model_id is required when --push_to_hub is set.")

    os.makedirs(args.output, exist_ok=True)

    logger.info(f"Loading checkpoint from {args.input} ...")
    checkpoint = torch.load(args.input, map_location="cpu")

    if args.classification:
        model = _convert_classification(checkpoint, args.output, args.safe_serialization)
    else:
        model = _convert_backbone(checkpoint, args.output, args.safe_serialization)

    # Preserve training metadata alongside the model weights so nothing is lost.
    training_meta = {
        k: checkpoint[k]
        for k in ("global_step", "best_val_loss", "optimizer_state_dict",
                  "lr_scheduler_state_dict", "patience_count",
                  "pretraining_vocab_sizes", "best_val_accuracy")
        if k in checkpoint
    }
    if training_meta:
        meta_path = os.path.join(args.output, "training_state.pt")
        torch.save(training_meta, meta_path)
        logger.info(f"Training metadata saved to {meta_path}")

    if args.push_to_hub:
        token = args.hub_token or os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "HuggingFace token required for Hub push. "
                "Pass --hub_token or set the HF_TOKEN environment variable."
            )
        logger.info(f"Pushing to Hub: {args.hub_model_id} ...")
        model.push_to_hub(args.hub_model_id, token=token)
        logger.info("Model pushed to Hub successfully.")


if __name__ == "__main__":
    main()
