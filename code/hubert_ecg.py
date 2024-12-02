import torch
import torch.nn as nn
from transformers import HubertConfig, HubertModel
from typing import List

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
