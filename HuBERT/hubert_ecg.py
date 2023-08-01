import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import copy
from transformers import HubertPreTrainedModel, HubertConfig
from transformers.models.hubert.modeling_hubert import HubertFeatureEncoder, HubertFeatureProjection, HubertEncoder, HubertEncoderStableLayerNorm
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from typing import Optional, Tuple, Union

from dataset import ECGDatasetSelfSupervised as ECGDataset
import pandas as pd

class HuBERT(HubertPreTrainedModel):
    def __init__(self, config: HubertConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config)
        else:
            self.encoder = HubertEncoder(config)

        #final projection layer
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

        #embedding for codebook
        self.label_embedding = nn.Embedding(config.vocab_size, config.classifier_proj_size)

        # Initialize weights and apply final processing
        self.post_init()

    def logits(self, transformer_output: torch.Tensor) -> torch.Tensor:
        #takes (B, T, D)
        projected_output = self.final_proj(transformer_output) #(B, T, C)
        logits = torch.cosine_similarity(
            projected_output.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1 #returns (BS, T, V)

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
            return (hidden_states,) + encoder_outputs[1:], mask_time_indices

        final_dict = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        final_dict["mask_time_indices"] = mask_time_indices
        return final_dict

def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


# if __name__ == "__main__":

#     config = HubertConfig(
#         vocab_size = 100,
#         hidden_size = 768,
#         num_hidden_layers = 12,
#         num_attention_heads = 12,
#         intermediate_size = 3072,
#         hidden_act = 'gelu',
#         mask_time_prob = 0.5, 
#         mask_time_length = 10,
#         mask_time_min_masks = 2,
#         classifier_proj_size = 256,
#     ) # + other default params

#     hubert = HuBERT(config)

#     dataframe = pd.read_csv("/data/ECG_AF/train_self_supervised_processed.csv")

#     record = dataframe.iloc[0]

#     data = np.load("/data/ECG_AF/train_self_supervised/" + record.filename)
#     data = data[:, :2500]
#     data = np.concatenate(data)
#     data = torch.from_numpy(data).float()
#     data = data.unsqueeze(0) #adding batch dimension

#     # print(data.shape) # (1, 12*2500)

#     # By passing no mask_time_indices to forward, if the model is in training mode and the mask_prob > 0, then masking will be applied

#     out_encoder = hubert(data, attention_mask = None, output_attentions=True, output_hidden_states=True, return_dict=True)

#     print("out_encoder keys: ", out_encoder.keys()) # dict_keys(['last_hidden_state', 'hidden_states', 'attentions', 'mask_time_indices'])
#     print("out_encoder.last_hidden_state.shape: ", out_encoder['last_hidden_state'].shape) # (BS, F, D)
#     print("out_encoder.hidden_states.length: ", len(out_encoder['hidden_states'])) # num_encoder_layers + 1
#     print("out_encoder.hidden_states[0].shape: ", out_encoder['hidden_states'][0].shape) # (BS, F, D)
#     # print("out_enoder.attentions.length: ", len(out_encoder['attentions'])) # num_encoder_layers
#     # print("out_encoder.attentions[0].shape: ", out_encoder['attentions'][0].shape) # (BS, N, F, F)
#     print("out_encoder.mask_time_indices.shape: ", out_encoder['mask_time_indices'].shape) # (BS, F), bool

#     logits = hubert.logits(out_encoder['last_hidden_state']) #(BS, F, V)

#     print("Logists.shape: ", logits.shape)

#     labels = torch.randint(0, config.vocab_size, (1, 93)) # (BS, F), values ranging from 0 to vocab_size-1

#     print("Labels.shape: ", labels.shape)

#     mask = out_encoder['mask_time_indices']

#     masked_loss = F.cross_entropy(logits[mask], labels[mask])

#     print("Masked loss: ", masked_loss)

#     unmasked_loss = F.cross_entropy(logits[~mask], labels[~mask])

#     print("Unmasked loss: ", unmasked_loss)

#     loss = 0.5 * masked_loss + 0.5 * unmasked_loss

#     print("Loss: ", loss)
