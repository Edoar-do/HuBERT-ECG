from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, List

from .configuration import HuBERTECGClassificationConfig
from .modeling import HuBERTECG, _remap_pos_conv_keys, _upgrade_config


class _ActivationFunction(nn.Module):
    _MAP = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }

    def __init__(self, name: str):
        super().__init__()
        if name not in self._MAP:
            raise ValueError(
                f"Unsupported activation '{name}'. "
                f"Choose from: {list(self._MAP)}"
            )
        self.act = self._MAP[name]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)


class HuBERTECGForClassification(PreTrainedModel):
    """
    HuBERT-ECG with a classification (or regression) head on top.

    Inherits from :class:`~transformers.PreTrainedModel`, enabling::

        # Load from HuggingFace Hub
        model = HuBERTECGForClassification.from_pretrained("Edoardo-BS/hubert-ecg-base-af164")

        # Load from a local HF-format directory
        model = HuBERTECGForClassification.from_pretrained("./my_finetuned/")

        # Save to disk (pytorch_model.bin by default)
        model.save_pretrained("./output/")

        # Save as safetensors (opt-in)
        model.save_pretrained("./output/", safe_serialization=True)

    For legacy ``.pt`` checkpoints use :meth:`from_pretrained_legacy`.

    The classification head has three variants controlled by config:

    * ``use_label_embedding=True``  — label-embedding cosine-similarity head
    * ``classifier_hidden_size=None`` — linear head (default)
    * ``classifier_hidden_size=N``  — two-layer MLP head with activation

    All existing freezing helpers are preserved from the original implementation.
    """

    config_class = HuBERTECGClassificationConfig
    base_model_prefix = "hubert_ecg"
    # Pre-training-only keys: present in backbone checkpoints but deleted here.
    _keys_to_ignore_on_load_missing = [r"final_proj", r"label_embedding"]

    def __init__(self, config: HuBERTECGClassificationConfig):
        super().__init__(config)

        # Build backbone; pre-training heads are deleted immediately.
        self.hubert_ecg = HuBERTECG(config)
        del self.hubert_ecg.final_proj
        del self.hubert_ecg.label_embedding

        self.classifier_dropout = nn.Dropout(config.classifier_dropout_prob)
        activation = _ActivationFunction(config.activation)

        if config.use_label_embedding:
            self.label_embedding = nn.Embedding(config.num_labels, config.hidden_size)
            self.classifier = None
        elif config.classifier_hidden_size is None:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.label_embedding = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.classifier_hidden_size),
                activation,
                nn.Linear(config.classifier_hidden_size, config.num_labels),
            )
            self.label_embedding = None

        # Required by PreTrainedModel: initialises weights and applies final
        # processing (tied embeddings, etc.).
        self.post_init()

    # ------------------------------------------------------------------
    # Helpers (unchanged from original HuBERTForECGClassification)
    # ------------------------------------------------------------------

    def set_feature_extractor_trainable(self, trainable: bool) -> None:
        """Set the convolutional feature extractor as (un)trainable."""
        self.hubert_ecg.feature_extractor.requires_grad_(trainable)

    def set_transformer_blocks_trainable(self, n_blocks: int) -> None:
        """Unfreeze only the last ``n_blocks`` transformer encoder layers."""
        assert 0 <= n_blocks <= self.config.num_hidden_layers, (
            f"n_blocks ({n_blocks}) must be in [0, {self.config.num_hidden_layers}]"
        )
        self.hubert_ecg.encoder.requires_grad_(False)
        for i in range(1, n_blocks + 1):
            self.hubert_ecg.encoder.layers[-i].requires_grad_(True)

    def _get_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        """Compute logits from the pooled encoder representation."""
        if self.config.use_label_embedding:
            return torch.cosine_similarity(
                pooled.unsqueeze(1),
                self.label_embedding.weight.unsqueeze(0),
                dim=-1,
            )
        return self.classifier(pooled)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        """
        Args:
            input_values: ``(B, T)`` raw ECG waveform tensor (12 leads flattened).
            attention_mask: ``(B, T)`` binary mask; 1 = valid, 0 = padding.
            labels: Ground-truth labels. Shape depends on ``task_type``:
                * ``multi_label``: ``(B, num_labels)`` float
                * ``multi_class``: ``(B,)`` long
                * ``regression``: ``(B,)`` float
                When provided, the loss is computed and returned.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a :class:`~transformers.modeling_outputs.SequenceClassifierOutput`.

        Returns:
            :class:`~transformers.modeling_outputs.SequenceClassifierOutput` with
            optional ``loss``, ``logits``, ``hidden_states``, ``attentions``.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        encodings = self.hubert_ecg(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden = encodings.last_hidden_state  # (B, T, D)

        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            padding_mask = self.hubert_ecg._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )
            hidden[~padding_mask] = 0.0
            pooled = hidden.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        pooled = self.classifier_dropout(pooled)
        logits = self._get_logits(pooled)  # (B, num_labels)

        loss = None
        if labels is not None:
            if self.config.task_type == "multi_label":
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            elif self.config.task_type == "multi_class":
                loss = F.cross_entropy(logits, labels.long())
            else:  # regression
                loss = F.mse_loss(logits.squeeze(-1), labels.float())

        if not return_dict:
            output = (logits,) + (
                encodings.hidden_states,
                encodings.attentions,
            )
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encodings.hidden_states,
            attentions=encodings.attentions,
        )

    # ------------------------------------------------------------------
    # Legacy loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_legacy(
        cls,
        pt_path: str,
        map_location: str = "cpu",
    ) -> "HuBERTECGForClassification":
        """
        Load a fine-tuned model from the old ``torch.save`` checkpoint format::

            {
                "model_state_dict": ...,
                "model_config": ...,
                "finetuning_vocab_size": N,
                "linear": True/False,
                "use_label_embedding": True/False,
                ...
            }

        New checkpoints produced by ``scripts/finetune.py`` use the standard
        HuggingFace :meth:`~transformers.PreTrainedModel.save_pretrained` format
        and do **not** need this method.

        Args:
            pt_path: Path to the legacy ``.pt`` file.
            map_location: PyTorch device string (default ``"cpu"``).

        Returns:
            Loaded :class:`HuBERTECGForClassification` instance.
        """
        checkpoint = torch.load(pt_path, map_location=map_location)
        backbone_config = _upgrade_config(checkpoint["model_config"])
        state = _remap_pos_conv_keys(checkpoint["model_state_dict"])

        num_labels = checkpoint.get("finetuning_vocab_size", 2)
        use_label_embedding = checkpoint.get("use_label_embedding", False)
        is_linear = checkpoint.get("linear", True)

        classifier_hidden_size = None
        if not is_linear and not use_label_embedding:
            # Infer hidden size from the MLP intermediate weight shape
            for k in state:
                if "classifier" in k and k.endswith(".0.weight"):
                    classifier_hidden_size = state[k].shape[0]
                    break

        cls_config = HuBERTECGClassificationConfig(
            num_labels=num_labels,
            classifier_hidden_size=classifier_hidden_size,
            use_label_embedding=use_label_embedding,
            **backbone_config.to_dict(),
        )

        model = cls(cls_config)
        model.load_state_dict(state, strict=False)
        return model
