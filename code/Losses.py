import torch
from torch import nn

class PatchRecLoss(nn.Module):
    def __init__(self, loss_type : str = 'mse'):
        super(PatchRecLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, batch_output, batch_real_patches, batch_indeces_masked_patches):
        
        assert batch_output.shape == batch_real_patches.shape
        assert batch_output.shape[0] == len(batch_indeces_masked_patches)

        reconstructed_patches = torch.cat([batch_output[n, i] for n, i in enumerate(batch_indeces_masked_patches)], dim=0)
        real_patches_to_compare = torch.cat([batch_real_patches[n, i] for n, i in enumerate(batch_indeces_masked_patches)], dim=0)

        if len(reconstructed_patches) > 0 and len(real_patches_to_compare) > 0:
            # Compute the MSE loss between the reconstructed and original patches         
            return torch.nn.functional.mse_loss(reconstructed_patches, real_patches_to_compare), batch_output, batch_real_patches
        else:
            return torch.nn.functional.mse_loss(batch_output, batch_real_patches), batch_output, batch_real_patches
