from __future__ import annotations

import torch
import torch.nn as nn
from transformers import HubertConfig, HubertModel
from transformers.models.hubert.modeling_hubert import _compute_mask_indices
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Optional, Union, Tuple

from .configuration import HuBERTECGConfig

#### Note: in order to pre-train HuBERT-ECG we have to slightly modify the HuBERT source
#### so that forward() (and _mask_hidden_states) return mask_time_indices.
#### We achieve this by overriding those methods below.


def _remap_pos_conv_keys(state_dict: dict) -> dict:
    """
    Remap positional-conv weight keys across transformers library versions.

    Older checkpoints store the weight-normalised parameters as
    ``parametrizations.weight.original0 / original1``; newer versions use
    ``weight_g / weight_v``.  This helper normalises to the current convention
    so that ``load_state_dict`` never fails due to a version mismatch.
    """
    out = {}
    for k, v in state_dict.items():
        if k.endswith("parametrizations.weight.original0"):
            k = k.replace("parametrizations.weight.original0", "weight_g")
        elif k.endswith("parametrizations.weight.original1"):
            k = k.replace("parametrizations.weight.original1", "weight_v")
        out[k] = v
    return out


def _upgrade_config(
    raw_config,
    vocab_sizes: Optional[List[int]] = None,
) -> HuBERTECGConfig:
    """
    Upgrade a bare :class:`~transformers.HubertConfig` (from old checkpoints)
    to a :class:`HuBERTECGConfig`.  If it is already a :class:`HuBERTECGConfig`
    it is returned unchanged.
    """
    if isinstance(raw_config, HuBERTECGConfig):
        return raw_config
    if isinstance(raw_config, HubertConfig):
        vocab_sizes = vocab_sizes or [100]
        return HuBERTECGConfig(
            ensemble_length=len(vocab_sizes),
            vocab_sizes=vocab_sizes,
            **raw_config.to_dict(),
        )
    raise TypeError(f"Unrecognised config type: {type(raw_config)}")


class HuBERTECG(HubertModel):
    """
    HuBERT-ECG backbone used during self-supervised pre-training.

    Adds ensemble projection heads (``final_proj``) and codebook embeddings
    (``label_embedding``) on top of the standard :class:`~transformers.HubertModel`.
    These heads are removed and replaced by a classification head in
    :class:`~hubert_ecg.modeling_classification.HuBERTECGForClassification`.
    """

    config_class = HuBERTECGConfig
    base_model_prefix = "hubert_ecg"

    def __init__(self, config: HuBERTECGConfig):
        super().__init__(config)
        self.config = config
        self.pretraining_vocab_sizes = config.vocab_sizes

        assert config.ensemble_length > 0 and config.ensemble_length == len(config.vocab_sizes), (
            f"ensemble_length {config.ensemble_length} must equal "
            f"len(vocab_sizes) {len(config.vocab_sizes)}"
        )

        # Projection layers mapping encoder output into codebook space
        self.final_proj = nn.ModuleList([
            nn.Linear(config.hidden_size, config.classifier_proj_size)
            for _ in range(config.ensemble_length)
        ])

        # Codebook embeddings — one per ensemble member
        self.label_embedding = nn.ModuleList([
            nn.Embedding(vocab_size, config.classifier_proj_size)
            for vocab_size in config.vocab_sizes
        ])

        assert len(self.final_proj) == len(self.label_embedding)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time and/or feature axes (SpecAugment).
        Overrides the parent to *also* return ``mask_time_indices`` so that the
        pre-training loss can be computed on masked positions only.
        """
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states, mask_time_indices

        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(
                mask_time_indices, device=hidden_states.device, dtype=torch.bool
            )
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(
                mask_feature_indices, device=hidden_states.device, dtype=torch.bool
            )
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states, mask_time_indices

    def logits(self, transformer_output: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute cosine-similarity logits against each codebook.

        Args:
            transformer_output: ``(B, T, D)`` encoder output.

        Returns:
            List of ``(B, T, V_k)`` tensors, one per ensemble member.
        """
        projected = [proj(transformer_output) for proj in self.final_proj]
        return [
            torch.cosine_similarity(
                p.unsqueeze(2),
                emb.weight.unsqueeze(0).unsqueeze(0),
                dim=-1,
            ) / 0.1
            for p, emb in zip(projected, self.label_embedding)
        ]

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask
            )

        hidden_states = self.feature_projection(extract_features)
        hidden_states, mask_time_indices = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]

        if not return_dict:
            if mask_time_indices is None:
                return (hidden_states,) + encoder_outputs[1:]
            else:
                return (hidden_states,) + encoder_outputs[1:] + (mask_time_indices,)

        final_dict = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        if mask_time_indices is not None:
            final_dict["mask_time_indices"] = mask_time_indices

        return final_dict

    # ------------------------------------------------------------------
    # Legacy loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_legacy(cls, pt_path: str, map_location: str = "cpu") -> "HuBERTECG":
        """
        Load a model from the old ``torch.save`` checkpoint format::

            {
                "model_state_dict": ...,
                "model_config": ...,
                "pretraining_vocab_sizes": [...],
                ...
            }

        New checkpoints produced by ``scripts/pretrain.py`` use the standard
        HuggingFace :meth:`~transformers.PreTrainedModel.save_pretrained` format
        and do **not** need this method.

        Args:
            pt_path: Path to the legacy ``.pt`` file.
            map_location: PyTorch device string (default ``"cpu"``).

        Returns:
            Loaded :class:`HuBERTECG` instance.
        """
        checkpoint = torch.load(pt_path, map_location=map_location)
        config = _upgrade_config(
            checkpoint["model_config"],
            checkpoint.get("pretraining_vocab_sizes"),
        )
        model = cls(config)
        state = _remap_pos_conv_keys(checkpoint["model_state_dict"])
        model.load_state_dict(state, strict=False)
        return model
