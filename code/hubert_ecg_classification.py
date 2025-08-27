import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from hubert_ecg import HuBERTECG, HuBERTECGConfig

class ActivationFunction(nn.Module):
    def __init__(self, activation : str):
        super(ActivationFunction, self).__init__()
        self.activation = activation
        
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activation function not supported')
    
    def forward(self, x):
        return self.act(x)

class HuBERTForECGClassification(nn.Module):

    def __init__(
        self,
        hubert_ecg : HuBERTECG,
        num_labels : int,
        classifier_hidden_size : int = None,
        activation : str = 'tanh',
        use_label_embedding : bool = False,
        classifier_dropout_prob : float = 0.1):
        super(HuBERTForECGClassification, self).__init__()
        self.hubert_ecg = hubert_ecg
        self.hubert_ecg.config.mask_time_prob = 0.0 # as we load pre-trained models that used to mask inputs, resetting masking probs prevents masking
        self.hubert_ecg.config.mask_feature_prob = 0.0 # as we load pre-trained models that used to mask inputs, resetting masking probs prevents masking
        
        self.num_labels = num_labels
        self.config = self.hubert_ecg.config
        self.classifier_hidden_size = classifier_hidden_size
        self.activation = ActivationFunction(activation)
        self.use_label_embedding = use_label_embedding 
        self.classifier_dropout = nn.Dropout(classifier_dropout_prob)
        
        del self.hubert_ecg.label_embedding # not needed
        del self.hubert_ecg.final_proj # not needed
        
        if use_label_embedding: # for classification only
            self.label_embedding = nn.Embedding(num_labels, self.config.hidden_size) 
        else:
            if classifier_hidden_size is None: # no hidden layer
                self.classifier = nn.Linear(self.config.hidden_size, num_labels)
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(self.config.hidden_size, classifier_hidden_size),
                    self.activation,
                    nn.Linear(classifier_hidden_size, num_labels)
                )
        
    def set_feature_extractor_trainable(self, trainable : bool):
        '''Sets as (un)trainable the convolutional feature extractor of HuBERT-ECG'''
        self.hubert_ecg.feature_extractor.requires_grad_(trainable)
    
    def set_transformer_blocks_trainable(self, n_blocks : int):
        ''' Makes trainable only the last `n_blocks` of HuBERT-ECG transformer encoder'''
        
        assert n_blocks >= 0, f"n_blocks (inserted {n_blocks}) should be >= 0"
        assert n_blocks <= self.hubert_ecg.config.num_hidden_layers, f"n_blocks ({n_blocks}) should be <= {self.hubert_ecg.config.num_hidden_layers}"
        
        self.hubert_ecg.encoder.requires_grad_(False)
        for i in range(1, n_blocks+1):
            self.hubert_ecg.encoder.layers[-i].requires_grad_(True)
                
    def get_logits(self, pooled_output : torch.Tensor):
        '''Computes cosine similary between transfomer pooled output, referred to as input representation, and look-up embedding matrix, that is a dense representation of labels.
        In: pooled_output: (B, C) tensor
        Out: (B, num_labels) tensor of similarities/logits to be sigmoided and used in BCE loss
        '''
        logits = torch.cosine_similarity(pooled_output.unsqueeze(1), self.label_embedding.weight.unsqueeze(0), dim=-1)
        return logits
            
    def forward(
        self,
        x: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Union[Tuple, BaseModelOutput]]:
        
        return_dict = return_dict if return_dict is not None else self.hubert_ecg.config.use_return_dict
        output_hidden_states = True if self.hubert_ecg.config.use_weighted_layer_sum else output_hidden_states
               
        encodings = self.hubert_ecg(
                x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            ) 

        if return_dict:
            x = encodings.last_hidden_state
        else:
            x = encodings[0]
        
        if attention_mask is None:
            x = x.mean(dim=1) 
        else:
            padding_mask = self.hubert_ecg._get_feature_vector_attention_mask(x.shape[1], attention_mask)
            x[~padding_mask] = 0.0
            x = x.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            
        x = self.classifier_dropout(x)
        
        logits = self.get_logits(x) if self.use_label_embedding else self.classifier(x)
        
        if return_dict:
            encodings["logits"] = logits
            return encodings
        else:
            return (logits, encodings)
