import torch
import torch.nn as nn
from transformers import HubertConfig, HubertModel
from typing import List

class HuBERTECG(HubertModel):
    
    config_class = HubertConfig
    
    def __init__(self, config: HubertConfig, ensemble_length : int = 1, vocab_sizes : List[int] = [100]):
        super().__init__(config)
        self.config = config

        self.pretraining_vocab_sizes = vocab_sizes if type(vocab_sizes) == list else [vocab_sizes] # if int is passed
            
        assert ensemble_length > 0 and ensemble_length == len(vocab_sizes), f"ensemble_length {ensemble_length} must be equal to len(vocab_sizes) {len(vocab_sizes)}"

        #final projection layer to map encodings into the space of the codebook
        self.final_proj = nn.ModuleList([nn.Linear(config.hidden_size, config.classifier_proj_size) for _ in range(ensemble_length)])

        #embedding for codebooks
        self.label_embedding = nn.ModuleList([nn.Embedding(vocab_size, config.classifier_proj_size) for vocab_size in vocab_sizes])
        
        assert len(self.final_proj) == len(self.label_embedding), f"final_proj and label_embedding must have the same length"
        
    def logits(self, transformer_output: torch.Tensor) -> torch.Tensor:
        #takes (B, T, D)
        
        # compute a projected output for each ensamble
        projected_outputs = [final_projection(transformer_output) for final_projection in self.final_proj]
        
        ensamble_logits = [torch.cosine_similarity(
            projected_output.unsqueeze(2),
            label_emb.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        ) / 0.1 for projected_output, label_emb in zip(projected_outputs, self.label_embedding)]
        
        return ensamble_logits #returns [(BS, T, V)] * ensemble_length
