
import torch
from torch import nn
import math
import numpy as np
from torch.cuda.amp import autocast

###<----- Start of encoding stack ----->###

class Patcher(nn.Module):
    def __init__(self, patch_size : tuple):
        super(Patcher, self).__init__()
        self.h, self.w = patch_size
    
    def forward(self, ecg_data : torch.Tensor):
        # takes (batch_size, 12, 5000)
        bs, n_leads, window = ecg_data.shape
        assert n_leads % self.h == 0
        assert window % self.w == 0
        # return (batch_size, n_patches, patch_height, patch_width)
        return ecg_data.unfold(1, self.h, self.h).unfold(2, self.w, self.w).reshape(bs, -1, self.h, self.w)


class Masker(nn.Module):
    def __init__(self, mask_token, mask_perc):
        super(Masker, self).__init__()
        self.mask_token = mask_token
        self.mask_perc = mask_perc

    def masking(self, batch_patches : torch.Tensor, mask_token):
        batch_mask_indexer = np.random.choice([True, False], size=batch_patches.shape[0], p=[0.85, .15])       
        batch_indeces = []
        for b, to_be_masked in enumerate(batch_mask_indexer):
            if to_be_masked: #some patches of the b instance will probably be masked
                patches_mask_indexer = torch.from_numpy(np.random.choice([True, False], 
                    size=batch_patches.shape[1], p=[self.mask_perc, 1-self.mask_perc])).cuda()
                #mask the patches of the b instance where corresponding patches_mask_indexer is True
                batch_patches[b] = torch.where(patches_mask_indexer.unsqueeze(1).unsqueeze(2), mask_token, batch_patches[b])
                #save the indeces of the patches that have been masked under the form of a list
                batch_indeces.append(torch.argwhere(patches_mask_indexer).squeeze(1).tolist())
            else:
                batch_indeces.append([])
        return batch_patches, batch_indeces

    def forward(self, batch_patches : torch.Tensor):
        #takes (batch_size, n_patches, patch_height, patch_width)
        #plus masked patches (bs, n_patches, patch_heitght, patch_width)
        #plus list bs long containing lists of indeces of patches that have been masked per instance
        return self.masking(batch_patches, self.mask_token)


class SimpleLinearEmbedder(nn.Module):
    ''' Simple linear projector as embedder of the ECG patches. Like ViTs.
    Works on both 1D and 2D patches'''
    def __init__(self, n_patches, patch_height, patch_width, embedding_size):
        super(SimpleLinearEmbedder, self).__init__()
        self.n_patches = n_patches
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.linear_embedding = nn.Linear(patch_height * patch_width, embedding_size, bias=False)
        self.name = 'linear projector'
    
    def forward(self, x):
        #(batch_size, n_patches, patch_height, patch_width)
        x = x.view(-1, self.n_patches, self.patch_height * self.patch_width) # like concatenating every row in each patch so that each new patch has shape (1, patch_width * patch_height)
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
        if patch_height == 1:
            self.kernels = [(1, 14), (1, 10)]
            self.strides = [2, 1]
        elif patch_height == 2 or patch_height == 3:
            self.kernels = [(2, 14), (2, 10)]
            self.strides = [(1, 2), (1, 1)]
        elif patch_height == 4 or patch_height == 6:
            self.kernels = [(3, 14), (3, 10)]
            self.strides = [(1, 2), (1, 1)]
        self.conv_embedder = nn.Sequential( 
            nn.Conv2d(n_patches, 64, self.kernels[0], self.strides[0], padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, self.kernels[0], self.strides[0], padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, self.kernels[1], self.strides[0], padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, self.kernels[1], self.strides[0], padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, self.kernels[1], self.strides[1], padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_patches, self.kernels[1], self.strides[1], padding=0),
            nn.BatchNorm2d(n_patches),
            nn.ReLU(inplace=True)).cuda()
        
        self.linear_projection = nn.Linear(self.get_output_shape(), embedding_size, bias=False)
        # remove conv_embedder from device
        self.conv_embedder.cpu()

    def get_output_shape(self):
        #get output shape of conv_embedder from a random tensor with shape (bs, n_patches, patch_height, patch_width)
        #this method is called only once in the init methhod, which is called only once in the whole training process, i.e. when the model is created
        bs = 1
        x = torch.rand((bs, self.n_patches, self.patch_height, self.patch_width)).cuda().to(non_blocking=True)
        x = self.conv_embedder(x)
        x.detach().cpu()
        
        return x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.conv_embedder(x) #(batch_size, n_patches, h_out == x.shape[2], w_out == x.shape[3])
        #Linear projection to map cnn outputs size to embedding size
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]) #(batch_size, n_patches, h_out x w_out)
        x = self.linear_projection (x) #(batch_size, n_patches , embedding_size == d_model)
        return x 
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, n_patches, p_dropout, max_length = 5000, learnable = False):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.name = 'positional encoding 1d'
        pe = torch.zeros((n_patches, d_model))
        pos = torch.arange(0, n_patches).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2) * (math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos(pos / denominator)
        if not learnable:
            self.register_buffer("pe", pe)
        self.pe = pe

    def forward(self, x):
        #x has shape (batch_size, n_patches+1, d_model)
        x = x + self.pe.cuda().to(non_blocking=True)
        return self.dropout(x) #exits (batch_size, n_patches+1, d_model)
    
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, n_patches, p_dropout, max_length = 5000, learnable = False):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.name = 'positional encoding 2d'
        self.row_pe = self.get_pos_encoding(n_patches, d_model)
        self.col_pe = self.get_pos_encoding(d_model, n_patches).transpose(0, 1)
        pe = self.row_pe + self.col_pe
        if not learnable:
            self.register_buffer('pe', pe)
        self.pe = pe
    
    def forward(self, x):
        x = x + self.pe.cuda().to(non_blocking=True)
        return self.dropout(x) #exits (batch_size, n_patches, d_model)

    def get_pos_encoding(self, n_patches, d_model):
            pe = torch.zeros((n_patches, d_model))
            pos = torch.arange(0, n_patches).unsqueeze(1)
            denominator = torch.exp(torch.arange(0, d_model, 2) * (math.log(10000.0)/d_model))
            pe[:, 0::2] = torch.sin(pos / denominator)
            pe[:, 1::2] = torch.cos(pos / denominator)
            return pe

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
        # takes (batch_size, n_patches, d_model)
        return self.transformer_encoder(x) # outputs (batch_size, n_patches , d_model == embedding_size)
    
###<----- End of encoding stakck ----->###
    
###<----- Start of decoding stack ---->###    
class ReversedEncoderLayer(nn.Module):
    ''' Reversed encoder layer'''
    def __init__(self, d_model, p_dropout, dim_ff, n_heads):
        super(ReversedEncoderLayer, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.Linear(dim_ff, d_model),
            nn.ReLU())
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, p_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)
        self.name = 'reversed encoder layer'

    def forward(self, x):
        #x has shape (batch_size, n_patches, d_model)
        #add & norm --> feed forward --> add & norm --> self attention
        x = self.norm1(x + self.dropout(x))
        x = self.ff(x) 
        x = self.norm2(x + self.dropout(x))
        x, attn_weights = self.self_attention(x, x, x)
        return x #exits (batch_size, n_patches, d_model)

class ReversedTransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, p_dropout, dim_ff, n_heads):
        super(ReversedTransformerEncoder, self).__init__()
        self.n_layers = n_layers        
        modules = nn.ModuleList([ReversedEncoderLayer(d_model, p_dropout, dim_ff, n_heads) for _ in range(n_layers)])
        self.layers = nn.Sequential(*modules)
        self.name = 'reversed transformer encoder'        
    
    def forward(self, x):
        #takes (batch_size, n_patches, d_model)
        return self.layers(x) #exits (batch_size, n_patches, d_model)

class FullyConvDecoder(nn.Module):
    def __init__(self, n_patches, d_model, patch_height, patch_width):
        super(FullyConvDecoder, self).__init__()
        self.n_patches = n_patches
        self.d_model = d_model
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.name = 'fully convolutional decoder'

        # ? could be necessary to improve this decoder architecture and hyperparams. This is just a first try

        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(n_patches, n_patches, kernel_size=3, stride=2),
            nn.BatchNorm2d(n_patches),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_patches, n_patches, kernel_size=3, stride=2),
            nn.BatchNorm2d(n_patches),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_patches, n_patches, kernel_size=3, stride=2),
            nn.BatchNorm2d(n_patches),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_patches, n_patches, kernel_size=3, stride=2),
            nn.BatchNorm2d(n_patches),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_patches, n_patches, kernel_size=3, stride=2),
            nn.BatchNorm2d(n_patches),
            nn.ReLU(inplace=True)
        ).cuda()

        self.w_proj = nn.Linear(self.get_output_shape()[0], patch_width, bias=False)
        self.h_proj = nn.Linear(self.get_output_shape()[1], patch_height, bias=False)

        self.conv_decoder.cpu()

    def get_output_shape(self):
        x = torch.randn(1, self.n_patches, self.d_model).cuda().to(non_blocking=True)
        x = x.unsqueeze(2) #adding height dim so to make deconv2d feasible
        x = self.conv_decoder(x)
        x.detach().cpu()
        return x.shape[-1], x.shape[-2] #exits (conv_width, conv_height)
            

    def forward(self, x):
        # adding height dim so to make deconv2d feasible
        #takes (batch_size, n_patches, d_model)
        x = x.unsqueeze(2)
        x = self.conv_decoder(x) #exits (batch_size, n_patches, conv_height, conv_width)
        x = self.w_proj(x)# (batch_size, n_patches, conv_heigth, patch_width)
        x = x.permute(0, 1, 3, 2) # (batch_size, n_patches, patch_width, conv_heigth)
        x = self.h_proj(x)# (batch_size, n_patches, patch_width, patch_height)
        return x.permute(0, 1, 3, 2) # (batch_size, n_patches, patch_height, patch_width)

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

class MirroredDecoder(nn.Module):
    def __init__(self, d_model, n_layers, p_dropout, dim_ff, n_heads, embeddying_type, patch_height, patch_width, n_patches):
        super(MirroredDecoder, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.dim_ff = dim_ff
        self.n_heads = n_heads
        self.embedding_type = embeddying_type
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.n_patches = n_patches
        self.name = 'mirrored decoder'

        self.reversed_encoder = ReversedTransformerEncoder(n_layers, d_model, p_dropout, dim_ff, n_heads)
        if embeddying_type == 'linear':
            self.lin1 = nn.Linear(d_model, patch_width, bias=False)
            self.lin2 = nn.Linear(1, patch_height, bias=False)
        if embeddying_type == 'conv':
            self.conv_embed = FullyConvDecoder(n_patches, d_model, patch_height, patch_width)
        
    def forward(self, x):
        # takes (batch_size, n_patches, d_model)
        x = self.reversed_encoder(x)
        if self.embedding_type == 'linear':
            x = x.unsqueeze(2) # (batch_size, n_patches, 1, d_model)
            x = self.lin1(x) # (batch_size, n_patches, 1, patch_width)
            x = x.permute(0, 1, 3, 2) # (batch_size, n_patches, patch_width, 1)
            x = self.lin2(x) # (batch_size, n_patches, patch_width, patch_height)
            x = x.permute(0, 1, 3, 2) # (batch_size, n_patches, patch_height, patch_width)
        elif self.embedding_type == 'conv':
            x = self.conv_embed(x) # (batch_size, n_patches, patch_height, patch_width)
        return x

class NonAutoregressiveTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, p_dropout, n_heads, dim_ff):
        super(NonAutoregressiveTransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.p_dropout = p_dropout
        self.dim_ff = dim_ff
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p_dropout)

        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(dim_ff, d_model)
        )
    
    def forward(self, encoder_output, pos_encodings):
        # encoder_output is (batch_size, n_patches, d_model)
        # pos_encodings is (batch_size, n_patches, d_model)

        attn_output = self.attn1(pos_encodings, pos_encodings, pos_encodings)[0] # (batch_size, n_patches, d_model)
        query = self.norm1(pos_encodings + self.dropout(attn_output)) # (batch_size, n_patches, d_model)
        cross_attn_output = self.cross_attn(query, encoder_output, encoder_output)[0] # (batch_size, n_patches, d_model)
        out = self.norm2(query + self.dropout(cross_attn_output)) # (batch_size, n_patches, d_model)
        out = self.ff(out) # (batch_size, n_patches, d_model)
        out = self.norm3(out + self.dropout(out)) # (batch_size, n_patches, d_model)
        return out

class NonAutoregressiveTransformerDecoder(nn.Module):
    def __init__(self, d_model, n_patches, n_decoding_layers, p_dropout, dim_ff, n_heads, patch_height, patch_width):
        super(NonAutoregressiveTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.n_patches = n_patches
        self.n_decoding_layers = n_decoding_layers
        self.p_dropout = p_dropout
        self.dim_ff = dim_ff
        self.n_heads = n_heads
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.name = 'non autoregressive transformer decoder'

        self.layers = nn.ModuleList([NonAutoregressiveTransformerDecoderLayer(d_model, p_dropout, n_heads, dim_ff) for _ in range(n_decoding_layers)])

        self.lin1 = nn.Linear(d_model, patch_width, bias=False)
        self.lin2 = nn.Linear(1, patch_height, bias=False)

    def forward(self, encoder_output, pos_encodings):
        # encoder_output is (batch_size, n_patches, d_model)
        # pos_encodings is (batch_size, n_patches, d_model)

        for layer in self.layers:
            encoder_output = layer(encoder_output, pos_encodings)

        # linear projections to get (batch_size, n_patches, patch_height, patch_width)
        x = encoder_output.unsqueeze(2) # (batch_size, n_patches, 1, d_model)
        x = self.lin1(x) # (batch_size, n_patches, 1, patch_width)
        x = x.permute(0, 1, 3, 2) # (batch_size, n_patches, patch_width, 1)
        x = self.lin2(x) # (batch_size, n_patches, patch_width, patch_height)
        x = x.permute(0, 1, 3, 2) # (batch_size, n_patches, patch_height, patch_width)
        return x

class FullModel(nn.Module):
    def __init__(self, embedding_type, patch_height, patch_width, n_patches, d_model, learnable_pos_enc, p_dropout, n_heads,
                 dim_ff, n_encoding_layers, decoding_type, pos_enc_type, hiddem_dim, n_classes):
        super(FullModel, self).__init__()
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
        
        # Embedder
        if embedding_type == 'linear':
            self.embedder = SimpleLinearEmbedder(n_patches, patch_height, patch_width, d_model)
        elif embedding_type == 'conv':
            self.embedder = ConvEmbedder(n_patches, patch_height, patch_width, d_model)

        #Positional Encoder
        if pos_enc_type == '1d':
            self.pe = PositionalEncoding1D(d_model, n_patches, p_dropout, learnable=learnable_pos_enc)
        elif pos_enc_type == '2d':
            self.pe = PositionalEncoding2D(d_model, n_patches, p_dropout, learnable=learnable_pos_enc)
        
        # Transformer encoder
        self.transformer_encoder = TransfomerEncoder(d_model, n_heads, dim_ff, p_dropout, n_encoding_layers)

        # Decoder
        if decoding_type == None: # This means supervised training. Not applicable to self-supervised MSM
            self.decoder = MLPClassifier(d_model, n_classes, p_dropout, hiddem_dim)
        elif decoding_type == 'fcn': # fully convolutional
            self.decoder = FullyConvDecoder(n_patches, d_model, patch_height, patch_width)
        # NOTE: from now on using less layers to decode would be interesting to try
        elif decoding_type == 'mirrored':
            self.decoder = MirroredDecoder(d_model, n_encoding_layers//2, p_dropout, dim_ff, n_heads, embedding_type, patch_height, patch_width, n_patches)
        elif decoding_type == 'nar':
            self.decoder = NonAutoregressiveTransformerDecoder(d_model, n_patches, n_encoding_layers, p_dropout, dim_ff, n_heads, patch_height, patch_width)
        elif decoding_type == 'encoder':
            # TODO: transformer encoder used as decoder as in paper "Spatiotemporal self-supervised representation learning for multi-lead ECG signals"
            pass

    @autocast()
    def forward(self, x, age, sex):
        embedding = self.embedder(x)
        pos_embedding = self.pe(embedding)
        cls = torch.zeros((pos_embedding.shape[0], 1, self.d_model)).cuda()
        pos_embedding_plus_cls = torch.cat([cls, pos_embedding], dim=1)
        transfomer_encoding = self.transformer_encoder(pos_embedding_plus_cls)
        
        if self.decoding_type == None: #supervised training
            out = self.decoder(transfomer_encoding[:, 0, :], age, sex) #pass vector of shape (batch_size, 1, d_model), (batch_size, 1), (batch_size, 1)
        elif self.decoding_type == 'nar':
            out = self.decoder(transfomer_encoding[:, 1:, :], pos_embedding) #pass vector of shape (batch_size, n_patches, d_model), (batch_size, n_patches, d_model)
        else: #decode/reconstruct all encodings except the cls token
            out = self.decoder(transfomer_encoding[:, 1:, :]) #vector of shape (batch_size, n_patches, patch_height, patch_width)
        return out