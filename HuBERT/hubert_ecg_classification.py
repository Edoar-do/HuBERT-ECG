import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from hubert_ecg import HuBERTECG

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
    def __init__(self, hubert_ecg : HuBERTECG, num_labels : int, classifier_hidden_size : int = None, activation : str = 'tanh'):
        super(HuBERTForECGClassification, self).__init__()
        self.hubert_ecg = hubert_ecg
        self.hubert_ecg.config.mask_time_prob = 0.0 # prevents masking
        self.hubert_ecg.config.mask_feature_prob = 0.0 # prevents masking
        
        # num_labels may be different from vocab_size when fine_tuning a pretrained hubert_ecg (in that case all modules are reused except embeddings, which could be freezed)
        # otherwise, it should (not necessarily) be equal to vocab_size for consistency
        self.num_labels = num_labels
        self.config = self.hubert_ecg.config
        self.classifier_hidden_size = classifier_hidden_size
        self.activation = ActivationFunction(activation)
        self.final_projection = nn.Linear(self.config.hidden_size, self.config.classifier_proj_size)    
        
        # if num_labels == 1 the task is supposed to be a regression
        if classifier_hidden_size is None: # no hidden layer
            self.classifier = nn.Linear(self.config.classifier_proj_size, num_labels)
        else:
            # mlp head with tanh activation like in ViT
            self.classifier = nn.Sequential(
                nn.Linear(self.config.classifier_proj_size, classifier_hidden_size),
                self.activation,
                nn.Linear(classifier_hidden_size, num_labels)
            )
        
        del self.hubert_ecg.label_embedding # not needed for classification
        del self.hubert_ecg.final_proj # not needed for classification
        
    def set_feature_extractor_trainable(self, trainable : bool):
        '''Set as (un)trainable the convolutional feature extractor of HuBERTECG'''
        self.hubert_ecg.feature_extractor.requires_grad_ = trainable
        
    # def set_pretraining_label_embeddings_trainable(self, trainable : bool):
    #     '''Set (un)trainable the label embedding of HuBERTECG'''
    #     for label_embedding in self.hubert_ecg.label_embeddings:
    #         label_embedding.requires_grad_ = trainable
        
    def set_base_model_trainable(self, trainable : bool):
        ''' Set every single parameter of HuBERTECG (un)trainable'''
        for param in self.hubert_ecg.parameters():
            param.requires_grad = trainable
    
    def set_transformer_blocks_trainable(self, trainable : bool, n_blocks : int = None):
        '''Set as (un)trainable `n_blocks` of the transformer encoder starting from the final and going backward. 
        If n_blocks is None, all blocks are set as (un)trainable'''
        if n_blocks is None:
            self.hubert_ecg.encoder.requires_grad_ = trainable
        else:
            assert n_blocks <= self.config.num_hidden_layers, f"n_blocks must be <= {self.config.num_hidden_layers}"
            for i in range(self.config.num_hidden_layers - n_blocks, self.config.num_hidden_layers):
                self.hubert_ecg.encoder.layers[i].requires_grad_ = trainable
            
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
        
        # takes (B, T) raw signal and returns logits with shape (B, num_labels) ready to be used in loss function + Hubert base model output        
        encodings = self.hubert_ecg(
                x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            ) # (B, T, D)
        
        x = self.final_projection(encodings.last_hidden_state) # (B, T, C)
        
        if attention_mask is None:
            x = x.mean(dim=1) # (B, C)
        else:
            padding_mask = self.hubert_ecg._get_feature_vector_attention_mask(x.shape[1], attention_mask)
            x[~padding_mask] = 0.0
            x = x.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        
        # (logits, hubert_output_dict)
        # (B, num_labels), hidden_states, attentions
        return self.classifier(x), encodings
        
        
