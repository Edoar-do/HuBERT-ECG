
import torch
from torch import nn
import math
import numpy as np
from torch.cuda.amp import autocast
from loguru import logger

### <---- patcher and masker ---> ###

class Patcher(nn.Module):
    def __init__(self, patch_size : tuple):
        super(Patcher, self).__init__()
        self.h, self.w = patch_size
    
    def forward(self, ecg_data : torch.Tensor):
        # takes (batch_size, 12, 5000)
        bs, n_leads, window = ecg_data.shape
        assert n_leads % self.h == 0
        assert window % self.w == 0

        out = ecg_data.unfold(1, self.h, self.h).unfold(2, self.w, self.w)
        return out.reshape(bs, -1, self.h, self.w), out.shape #(batch_size, n_patches, patch_height,  patch_width) + unfolded_shape necessary for visual_comparing()
    
class Masker(nn.Module):
    def __init__(self, mask_token, to_take_perc, mask_or_same_perc):
        super(Masker, self).__init__()
        self.mask_token = mask_token
        self.to_take_perc = to_take_perc
        self.mask_or_same_perc = mask_or_same_perc

    def patch_indexing(self, num_patches):
        #always select 15% of patches randomly
        n_patches_to_mask = int(num_patches * self.to_take_perc)
        random_indices = (torch.randperm(num_patches)[:n_patches_to_mask]).cuda()
        mask_or_same = random_indices.bernoulli(p=self.mask_or_same_perc)
        random_indices = random_indices * mask_or_same #mask out indices sampled as 'same'
        return random_indices[torch.nonzero(random_indices)]

    def roberta_masking(self, batch_patches):
        batch_indeces = []
        for instance in batch_patches:
            random_indices = self.patch_indexing(batch_patches.shape[1])
            instance[random_indices] = self.mask_token
            batch_indeces.append(random_indices.squeeze().cpu().tolist())            
        return batch_patches, batch_indeces

    def forward(self, batch_patches : torch.Tensor):
        #takes (batch_size, n_patches, patch_height, patch_width)
        #returns masked patches (bs, n_patches, patch_heitght, patch_width)
        #plus a list bs long containing lists of indeces of patches that have been masked per instance
        return self.roberta_masking(batch_patches)

###<----- Start of encoding stack ----->###

class SimpleLinearEmbedder(nn.Module):
    ''' Simple linear projector as embedder of the ECG patches. Like ViTs.
    Works on both 1D and 2D patches'''
    def __init__(self, n_patches, patch_height, patch_width, embedding_size):
        super(SimpleLinearEmbedder, self).__init__()
        self.n_patches = n_patches
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.linear_embedding = nn.Linear(patch_height * patch_width, embedding_size)
        self.name = 'linear projector'
    
    def forward(self, x):
        #if flattened (batch_size, n_patches, patch_width * patch_height)
        return self.linear_embedding(x) #outputs (batch_size, n_patches, embedding_size)
    
class ConvEmbedder(nn.Module):
    '''Convolutional Embedder to learn spatial information in the patches (independently and parallely across patches)'''
    def __init__(self, n_patches, patch_height, patch_width, embedding_size):
        super(ConvEmbedder, self).__init__()
        self.embedding_size = embedding_size
        self.n_patches = n_patches
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.embedding_size = embedding_size
        self.name = 'convolutional embedder'
        
        self.conv_embedder = nn.Sequential(
            nn.Conv1d(n_patches, n_patches, 15),
            nn.BatchNorm1d(n_patches),
            nn.GELU(),
            nn.Conv1d(n_patches, n_patches, 11),
            nn.BatchNorm1d(n_patches),
            nn.GELU(),
            nn.Conv1d(n_patches, n_patches, 9),
            nn.BatchNorm1d(n_patches),
            nn.GELU())
        
        self.linear_projection = nn.Linear(self.get_output_shape(), embedding_size)
        
        # remove conv_embedder from device
        # self.conv_embedder.cpu()

    def get_output_shape(self):
        #get output shape of conv_embedder from a random tensor with shape (bs, n_patches, patch_height, patch_width)
        #this method is called only once in the init methhod, which is called only once in the whole training process, i.e. when the model is created
        bs = 1
        x = torch.rand((bs, self.n_patches, self.patch_height * self.patch_width)).to(non_blocking=True)
        x = self.conv_embedder(x)
        x.detach().cpu()
        
        return x.shape[2]

    def forward(self, x):
        #if flattened (batch_size, num_patches, patch_height * patch_width)
        x = self.conv_embedder(x) #outputs (batch_size, n_patches, h_out == x.shape[2], w_out == x.shape[3])
        x = self.linear_projection (x) #(batch_size, n_patches , embedding_size == d_model)
        return x 
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, n_patches, p_dropout, max_length = 5000):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.name = 'positional encoding 1d'
        pe = torch.zeros((n_patches, d_model))
        pos = torch.arange(0, n_patches).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2) * (math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos(pos / denominator)
        self.pe = pe.cuda().to(non_blocking=True)

    def forward(self, x):
        #x has shape (batch_size, n_patches, d_model)
        x = x + self.pe
        return self.dropout(x) #exits (batch_size, n_patches, d_model)
    
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, n_patches, p_dropout, max_length = 5000):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.name = 'positional encoding 2d'
        self.row_pe = self.get_pos_encoding(n_patches, d_model)
        self.col_pe = self.get_pos_encoding(d_model, n_patches).transpose(0, 1)
        pe = self.row_pe + self.col_pe
        self.pe = pe.cuda().to(non_blocking=True)
    
    def forward(self, x):
        x = x + self.pe
        return self.dropout(x) #exits (batch_size, n_patches, d_model)

    def get_pos_encoding(self, n_patches, d_model):
            pe = torch.zeros((n_patches, d_model))
            pos = torch.arange(0, n_patches).unsqueeze(1)
            denominator = torch.exp(torch.arange(0, d_model, 2) * (math.log(10000.0)/d_model))
            pe[:, 0::2] = torch.sin(pos / denominator)
            pe[:, 1::2] = torch.cos(pos / denominator)
            return pe

class LearnablePositionalEncoding1D(nn.Module):
    def __init__(self, d_model, num_patches):
        super(LearnablePositionalEncoding1D, self).__init__()
        self.pe = nn.Parameter(torch.zeros(num_patches, d_model))
    def forward(self, embeddings):
        return embeddings + self.pe
    
class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, num_patches):
        super(LearnablePositionalEncoding2D, self).__init__()
        self.row_pe = nn.Parameter(torch.zeros(num_patches, d_model)) 
        self.col_pe = nn.Parameter(torch.zeros(d_model, num_patches).T)
        
    def forward(self, embeddings):
        return embeddings + self.row_pe + self.col_pe
 
    
class TransfomerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, p_dropout, n_encoding_layers):
        super(TransfomerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_ff, 
            dropout=p_dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_encoding_layers)
        self.name = 'transformer encoder'        

    def forward(self, x):
        # takes (batch_size, n_patches+1, d_model)
        return self.transformer_encoder(x) # outputs (batch_size, n_patches+1, d_model)
    
###<----- End of encoding stakck ----->###
    
###<----- Start of decoding stack ---->###

class EncoderAsDecoder(nn.Module):
    def __init__(self, d_model, p_dropout, dim_ff, n_heads, patch_height, patch_width, n_patches, n_decoding_layers):
        super(EncoderAsDecoder, self).__init__()
        self.d_model = d_model
        self.p_dropout = p_dropout
        self.dim_ff = dim_ff
        self.n_heads = n_heads
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.n_patches = n_patches
                
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = n_heads,
            dim_feedforward = dim_ff,
            dropout = p_dropout
        )
        
        self.decoder = nn.TransformerEncoder(self.decoder_layer, n_decoding_layers)
  
        self.lin1 = nn.Linear(d_model, patch_width*patch_height)
        self.name = "encoder as decoder"
    
    def forward(self, x):
        #takes (batch_size, n_patches, d_model)
        x = self.decoder(x) #(batch_size, n_patches, d_model)
        x = self.lin1(x) #(batch_size, n_patches, patch_height*patch_width)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, d_model, n_classes, p_dropout, hidden_dim):
        super(MLPClassifier, self).__init__()
        self.n_classes = n_classes
        # MLP head for classification. It takes the cls token from the output of the transformer encoder and classifies it. Not too deep
        self.classifier = nn.Sequential(
            nn.Linear(d_model+2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self.name = 'mlp classifier'

    def forward(self, x, age, sex):
        # takes (batch_size, d_model), (batch_size, 1), (batch_size, 1)
        # age = age.to(x.device, non_blocking=True)
        # sex = sex.to(x.device, non_blocking=True)
        x = torch.cat([x, age, sex], dim=1)
        return self.classifier(x) # (batch_size, n_classes), these are logits
   
class FullModel(nn.Module):
    def __init__(self, mask_token, to_take_perc, mask_or_same_perc, embedding_type, patch_height, patch_width, n_patches, d_model, learnable_pos_enc, p_dropout, n_heads,
                 dim_ff, n_encoding_layers, decoding_type, pos_enc_type, hiddem_dim, n_classes, dim_ff_decoding, n_decoding_heads, n_decoding_layers):
        super(FullModel, self).__init__()
        self.mask_token = mask_token
        self.to_take_perc = to_take_perc
        self.mask_or_same_perc = mask_or_same_perc
        self.embedding_type = embedding_type
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.n_patches = n_patches
        self.d_model = d_model
        self.learnable_pos_enc = learnable_pos_enc
        self.p_dropout = p_dropout
        self.n_heads = n_heads
        self.dim_ff = dim_ff
        self.n_encoding_layers = n_encoding_layers
        self.decoding_type = decoding_type
        self.pos_enc_type = pos_enc_type
        self.cls = nn.Parameter(torch.zeros((1, self.d_model)))
        
        self.patcher = Patcher((patch_height, patch_width))
        
        self.masker = Masker(mask_token, to_take_perc, mask_or_same_perc)
        
        # Embedder
        if embedding_type == 'linear':
            self.embedder = SimpleLinearEmbedder(n_patches, patch_height, patch_width, d_model)
        elif embedding_type == 'conv':
            self.embedder = ConvEmbedder(n_patches, patch_height, patch_width, d_model)

        #Positional Encoder
        if learnable_pos_enc:
            if pos_enc_type == '1d':
                self.pe = LearnablePositionalEncoding1D(d_model, n_patches)
            elif pos_enc_type == '2d':
                self.pe = LearnablePositionalEncoding2D(d_model, n_patches)
        else:
            if pos_enc_type == '1d':
                self.pe = PositionalEncoding1D(d_model, n_patches, p_dropout)
            elif pos_enc_type == '2d':
                self.pe = PositionalEncoding2D(d_model, n_patches, p_dropout)
            
        
        # Transformer encoder
        self.transformer_encoder = TransfomerEncoder(d_model, n_heads, dim_ff, p_dropout, n_encoding_layers)

        # Decoder
        if decoding_type == None: # This means supervised training. Not applicable to self-supervised MSM
            self.decoder = MLPClassifier(d_model, n_classes, p_dropout, hiddem_dim)        
        elif decoding_type == 'rev':
            self.decoder = EncoderAsDecoder(d_model, p_dropout, dim_ff, n_heads, patch_height, patch_width, n_patches, n_decoding_layers)
            
        self.initialize_weights()
        
    def initialize_weights(self):
        if self.decoding_type is not None: #when pre-training mask and cls token need to be initialized
            torch.nn.init.trunc_normal_(self.cls, std=0.02)
            
        for m in self.embedder.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                
        
        if self.learnable_pos_enc:
            if self.pos_enc_type == '1d':
                torch.nn.init.trunc_normal_(self.pe.pe, std=0.02)
            else:
                torch.nn.init.trunc_normal_(self.pe.row_pe, std=0.02)
                torch.nn.init.trunc_normal_(self.pe.col_pe, std=0.02)
        
        for m in self.transformer_encoder.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
                     
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
                
        logger.info("Model's weights initialized")

    @autocast()
    def forward(self, x, age, sex):
        patches, unfolded_shape = self.patcher(x)
        if self.decoding_type is not None:
            masked_patches, masked_indices = self.masker(patches.detach().clone())
            embedding = self.embedder(masked_patches.flatten(2))
        else:
            embedding = self.embedder(patches.flatten(2))
        
        pos_embedding = self.pe(embedding)
        cls_token = self.cls.expand(pos_embedding.shape[0], -1, -1)
        pos_embedding_plus_cls = torch.cat([cls_token, pos_embedding], dim=1)
        transformer_encodings = self.transformer_encoder(pos_embedding_plus_cls)
        
        if self.decoding_type == None: #supervised training
            out = self.decoder(transformer_encodings[:, 0, :], age, sex) #pass vector of shape (batch_size, 1, d_model), (batch_size, 1), (batch_size, 1)
            return out.unflatten(2, (self.patch_height, self.patch_width))
        else: #decode/reconstruct all encodings except the cls token
            out = self.decoder(transformer_encodings[:, 1:, :]) #vector of shape (batch_size, n_patches, patch_height, patch_width)
            return out.unflatten(2, (self.patch_height, self.patch_width)), patches, masked_indices, unfolded_shape

#<-- MAE like FAIR -->
    
# class MAE(nn.Module):
#     def __init__(self, num_patches, patch_height, patch_width, embedding_type, pos_enc_type, learnable_pos_enc, d_model, n_heads, dim_ff_ratio, n_encoding_layers, n_decoding_heads, n_decoding_layers):
#         super(MAE, self).__init__()
#         self.num_patches = num_patches
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.embedding_type = embedding_type
#         self.pos_enc_type = pos_enc_type
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.dim_ff_ratio = dim_ff_ratio
#         self.n_encoding_layers = n_encoding_layers
#         self.n_decoding_layers = n_decoding_layers
#         self.n_decoding_heads = n_decoding_heads
#         self.use_pe_in_decoding = use_pe_in_decoding
        
#         # Embedder
#         if embedding_type == 'linear':
#             self.embedder = SimpleLinearEmbedder(n_patches, patch_height, patch_width, d_model)
#         elif embedding_type == 'conv':
#             self.embedder = ConvEmbedder(n_patches, patch_height, patch_width, d_model)
            
#         #Positional Encoder
#         if learnable_pos_enc:
#             if pos_enc_type == '1d':
#                 self.pe = LearnablePositionalEncoding1D(d_model, n_patches)
#             elif pos_enc_type == '2d':
#                 self.pe = LearnablePositionalEncoding2D(d_model, n_patche)
#         else:
#             if pos_enc_type == '1d':
#                 self.pe = PositionalEncoding1D(d_model, n_patches, p_dropout)
#             elif pos_enc_type == '2d':
#                 self.pe = PositionalEncoding2D(d_model, n_patches, p_dropout)
        
#         #Encoder
#         self.encoder = TransfomerEncoder(d_model, n_heads, d_model * dim_ff_ratio, p_dropout, n_encoding_layers)
            
        
        