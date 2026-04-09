from __future__ import annotations

from typing import List, Optional

from transformers import HubertConfig


class HuBERTECGConfig(HubertConfig):
    """
    Configuration for HuBERT-ECG backbone (pre-training model).

    Extends :class:`~transformers.HubertConfig` with two extra fields that
    control the ensemble of codebook projection heads used during self-supervised
    pre-training.

    Args:
        ensemble_length: Number of K-means codebooks used in the ensemble.
            Must equal ``len(vocab_sizes)``.
        vocab_sizes: List of codebook sizes (one per ensemble member).
    """

    model_type = "hubert_ecg"

    def __init__(
        self,
        ensemble_length: int = 1,
        vocab_sizes: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ensemble_length = ensemble_length
        vocab_sizes = vocab_sizes or [100]
        self.vocab_sizes = vocab_sizes if isinstance(vocab_sizes, list) else [vocab_sizes]
        assert self.ensemble_length == len(self.vocab_sizes), (
            f"ensemble_length {self.ensemble_length} must equal "
            f"len(vocab_sizes) {len(self.vocab_sizes)}"
        )


class HuBERTECGClassificationConfig(HuBERTECGConfig):
    """
    Configuration for HuBERT-ECG fine-tuned for classification or regression.

    Extends :class:`HuBERTECGConfig` with parameters that fully describe the
    classification head, so that :meth:`~transformers.PreTrainedModel.from_pretrained`
    can reconstruct the entire model (backbone + head) from a single config.

    Args:
        num_labels: Number of output classes/labels.
        classifier_hidden_size: Hidden size of the MLP head. ``None`` → linear
            (single-layer) head.
        classifier_dropout_prob: Dropout probability applied before the head.
        activation: Activation function for the MLP head
            (``"tanh"``, ``"relu"``, ``"gelu"``, ``"sigmoid"``).
        use_label_embedding: If ``True``, use a learnable label-embedding
            matrix and predict via cosine similarity instead of a linear layer.
        task_type: One of ``"multi_label"``, ``"multi_class"``, ``"regression"``.
            Determines the loss function used during fine-tuning.
    """

    model_type = "hubert_ecg_classification"

    def __init__(
        self,
        num_labels: int = 2,
        classifier_hidden_size: Optional[int] = None,
        classifier_dropout_prob: float = 0.1,
        activation: str = "tanh",
        use_label_embedding: bool = False,
        task_type: str = "multi_label",
        **kwargs,
    ):
        # Pre-training ensemble heads are irrelevant for classification;
        # set minimal defaults so the parent validates correctly.
        kwargs.setdefault("ensemble_length", 1)
        kwargs.setdefault("vocab_sizes", [100])
        # Disable SpecAugment masking at config level — no mutation at runtime.
        kwargs.setdefault("mask_time_prob", 0.0)
        kwargs.setdefault("mask_feature_prob", 0.0)
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.classifier_hidden_size = classifier_hidden_size
        self.classifier_dropout_prob = classifier_dropout_prob
        self.activation = activation
        self.use_label_embedding = use_label_embedding
        self.task_type = task_type
