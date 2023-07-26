import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

class Hubert(nn.Module):
    def __init__(
        self,
        num_label_embeddings: int = 100,
        to_mask: bool = True,
        kernels_dims:tuple = (10, 3, 3, 3, 3, 2, 2),
        strides:tuple = (5, 2, 2, 2, 2, 2, 2),
        d_model: int = 768,
        nhead: int = 12,
        num_layers:int = 12,
        activation:str = 'gelu',
        dim_feedforward:int = 3072
        ):

        assert d_model % nhead == 0
        assert dim_feedforward % d_model == 0

        super(Hubert, self).__init__()

        ### hyper-params ###
        self.num_label_embeddings = num_label_embeddings #should be the number of clusters created by kmeans model
        self.to_mask = to_mask #only in pre-training, not fine-tuning
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.activation = activation
        self.dim_feedforward = dim_feedforward

        ### modules ###
        self.feature_extractor = FeatureExtractor(kernels_dims, strides)
        self.feature_projection = FeatureProjector(d_model)
        self.positional_embedding = PositionalEmbedding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = d_model,
                nhead = nhead,
                dim_feedforward = dim_feedforward,
                activation = activation,
                batch_first = True
            ),
            num_layers = num_layers
        )
        self.proj = nn.Linear(d_model, 256)

        ### tokens and embeddings ###
        self.mask_token = nn.Parameter(torch.FloatTensor(d_model).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x):
        '''Mask features `x` before the encoder'''
        mask = None
        if self.to_mask: #we mask only in pre-training, not fine-tuning
            mask = _compute_mask((x.size(0), x.size(1)), mask_prob=0.8, mask_length=10, device=x.device, min_masks=2)
            x[mask] = self.mask_token.to(x.dtype)
        return x, mask

    def encode(self, x, output_layer:int = None):
        print("X shape: ", x.shape)
        x = self.feature_extractor(x)
        print("feature extracted shape: ", x.shape)
        x = self.feature_projection(x.transpose(1, 2))
        print("feature projected shape: ", x.shape)
        x, mask = self.mask(x)
        print("features masked shape: ", x.shape)
        print("Mask shape: ", mask.shape)
        x = x + self.positional_embedding(x)
        print("features + pos shape: ", x.shape)
        x = self.dropout(self.norm(x))
        #output_layer useful when calling "predict" to compute representations for 2nd and 3rd train iterations
        #1st train iteration is None
        x = self.encoder(x, output_layer = output_layer) 
        print("transformer encodings shape: ", x.shape)
        return x, mask

    def logits(self, x):
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1

    def features(self, x, output_layer):
        ''' Return encoder's `output_layer`th features for a single ecg instance'''
        assert x.size(0) == 1 #one instance at a time
        features, _ = self.encode(x, output_layer)        
        return features.squeeze().cpu().numpy() #(cnn_out_shape, d_model)


    def forward(self, x):
        x, mask = self.encode(x)
        x = self.proj(x)
        print("final projection shape: ", x.shape)
        logits = self.logits(x)
        print("Logits shape: ", logits.shape)
        return logits, mask

class FeatureExtractor(nn.Module):
    def __init__(self, kernels_dims:tuple, strides:tuple):
        super(FeatureExtractor, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d((1 if i == 0 else 512), 512, k, s, bias=False) for i, (k, s) in enumerate(zip(kernels_dims, strides))]
        )
        self.norm = nn.GroupNorm(512, 512) #apply after first conv

    def forward(self, x):
        x = F.gelu(self.norm(self.convs[0](x)))
        for c in self.convs[1:]:
            x = F.gelu(c(x))
        return x

class FeatureProjector(nn.Module):
    def __init__(self, d_model):
        super(FeatureProjector, self).__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEmbedding, self).__init__()
        self.conv = nn.Conv1d(
            d_model, 
            d_model,
            kernel_size = 128, 
            padding = 128 // 2,
            groups = 16
        )
        self.conv == nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        x = F.gelu(x[:, :, :-1]) # ? why?
        return x.transpose(1, 2)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, src, mask = None, src_key_padding_mask = None, output_layer: int = None):
        output = src
        #specify output layer when calling "predict" in 2nd and 3rd train iteration
        for layer in self.layers[:output_layer]: 
            output = layer(output, src_mask = mask, src_key_padding_mask=src_key_padding_mask)
        return output

def _compute_mask(shape:tuple, mask_prob:float, mask_length:int, device:torch.device, min_masks:int=0):
    batch_size, sequence_length = shape

    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)

    uniform_distribution = torch.ones((batch_size, sequence_length - (mask_length - 1)), device=device)

    mask_indices = torch.multinomial(uniform_distribution, num_masked_spans)
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )

    mask_idxs = mask_indices + offsets

    return mask.scatter(1, mask_idxs, True)

if __name__ == "__main__":
    print("Testing Hubert...")
    hubert = Hubert()
    x = torch.randn(1, 1, 2500*12)
    logits, mask = hubert(x)

