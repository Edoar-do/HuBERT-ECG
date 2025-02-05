import torch
import torch.nn as nn
from transformers import HubertConfig, HubertModel
from transformers.models.hubert.modeling_hubert import _compute_mask_indices
from typing import List, Optional, Union, Tuple
from transformers.modeling_outputs import BaseModelOutput

#### Note: in order to pre-train HuBERT-ECG we have to slightly modify its source code so that the forward method (and the _mask_hidden_states as well)
#### returns the mask_time_indices. 
#### To do so, we have to overwrite the forward method of the HuBERT class adding the mask_time_indices to the return statement.

class HuBERTECGConfig(HubertConfig):
    
    model_type = "hubert_ecg"
    
    def __init__(self, ensemble_length: int = 1, vocab_sizes: List[int] = [100], **kwargs):
        super().__init__(**kwargs)
        self.ensemble_length = ensemble_length
        self.vocab_sizes = vocab_sizes if isinstance(vocab_sizes, list) else [vocab_sizes]

class HuBERTECG(HubertModel):
    
    config_class = HuBERTECGConfig
    
    def __init__(self, config: HuBERTECGConfig):
        super().__init__(config)
        self.config = config

        self.pretraining_vocab_sizes = config.vocab_sizes
            
        assert config.ensemble_length > 0 and config.ensemble_length == len(config.vocab_sizes), f"ensemble_length {config.ensemble_length} must be equal to len(vocab_sizes) {len(config.vocab_sizes)}"

        # final projection layer to map encodings into the space of the codebook
        self.final_proj = nn.ModuleList([nn.Linear(config.hidden_size, config.classifier_proj_size) for _ in range(config.ensemble_length)])

        # embedding for codebooks
        self.label_embedding = nn.ModuleList([nn.Embedding(vocab_size, config.classifier_proj_size) for vocab_size in config.vocab_sizes])
        
        assert len(self.final_proj) == len(self.label_embedding), f"final_proj and label_embedding must have the same length"

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states
        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()
        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0
        return hidden_states, mask_time_indices
        
    def logits(self, transformer_output: torch.Tensor) -> torch.Tensor:
        # takes (B, T, D)
        
        # compute a projected output for each ensemble
        projected_outputs = [final_projection(transformer_output) for final_projection in self.final_proj]
        
        ensemble_logits = [torch.cosine_similarity(
            projected_output.unsqueeze(2),
            label_emb.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        ) / 0.1 for projected_output, label_emb in zip(projected_outputs, self.label_embedding)]
        
        return ensemble_logits # returns [(BS, T, V)] * ensemble_length
    
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
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
        hidden_states = self.feature_projection(extract_features)
        hidden_states, mask_time_indices = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:] + mask_time_indices
        final_dict = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        final_dict["mask_time_indices"] = mask_time_indices
        return final_dict
