import torch
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import wandb
import numpy as np
from Model import FullModel, Patcher, Masker
from Configs import get_configs
from torch.cuda.amp import autocast, GradScaler
from Losses import PatchRecLoss
from loguru import logger


def visual_comparing(rec_patches, real_patches, unfolded_shape, masked_patches_indeces):
    '''Returns a figure showing the `reconstructed patches` and `real_patches` after their folding to original size.
    The figure highlights which parts of the original signal have been masked'''
    bs, res_height, res_width, patch_height, patch_width = unfolded_shape
    rec_patches = rec_patches.view(unfolded_shape)
    original_h = res_height * patch_height
    original_w = res_width * patch_width
    rec_patches = rec_patches.permute(0, 1, 3, 2, 4).contiguous().view(bs, original_h, original_w) #(bs, n_leads, window_size)
    
    real_patches = real_patches.view(unfolded_shape)
    real_patches = real_patches.permute(0, 1, 3, 2, 4).contiguous().view(bs, original_h, original_w) #(bs, n_leads, window_size)
    
    #subplots with original_h rows and 1 column. Each subplos shows the reconstructed lead and the real lead
    fig, axs = plt.subplots(original_h, 1, figsize=(20, 20), sharex=True)
    time_span = np.arange(0, original_w)
    for lead in range(original_h):
        axs[lead].plot(time_span, rec_patches[0, lead, :], label="reconstructed")
        axs[lead].plot(time_span, real_patches[0, lead, :], label="real")
    
    #color the regions corresponding to masked patches
    for idx in masked_patches_indeces:
        start = idx * patch_width
        end = (idx+1) * patch_width
        lead = end // original_w
        axs[lead] = plt.fill_between(time_span[start:end], rec_patches[0, lead, start:end], alpha = 0.5)
        
    plt.legend()
    return fig

def test_supervised(test_dataloader, model, loss_fn, *metrics):
    '''
    Test `model` performance on data retrieved by `test_dataloader` in a self-supervised fashion.
    Performance are measure according to `loss_fn` and additional `metrics`.
    Params:
        - `test_dataloader`: instance of torch.utils.data.DataLoader to fetch data from test set
        - `model`: the model to test
        - `loss_fn`: the loss function which evaluate the model with
        - `metrics`: any additional metric which evaluate the model with
    Returns:
        - test_loss
        - test labels
        - test probabilities
    '''
    
    configs = get_configs("./ECG_pretraining/code/configs.json")
    
    model.eval()
    losses = []
    lbls = []
    probs = []
    sigmoid = nn.Sigmoid()
    
    patcher = Patcher((configs['patch_height'], configs['patch_width'])).cuda() 
    
    age_min = test_dataloader.dataset.ecg_dataframe['age'].min()
    age_max = test_dataloader.dataset.ecg_dataframe['age'].max()

    logger.info("\t Testing model performance...")

    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_ecg_data, batch_age, batch_sex, batch_labels = tuple(zip(*batch))
        batch_ecg_data = torch.stack(batch_ecg_data, dim=0).cuda().to(dtype=torch.float)
        batch_ecg_data, _ = patcher(batch_ecg_data)
        #normalize age

        batch_age = torch.Tensor((batch_age - age_min)/(age_max - age_min)).cuda().to(dtype=torch.float) # normalized age, (batch_size, 1)
        batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float)# (batch_size, 1)
        with torch.no_grad():
            batch_labels = torch.stack(batch_labels, dim=0).cuda().to(dtype=torch.float)
            pred = model(batch_ecg_data, batch_age, batch_sex)
            loss = loss_fn(pred, batch_labels)
            probs = sigmoid(pred).detach().cpu().numpy()
            lbls = batch_labels.detach().cpu().numpy()
            lbls.append(lbls)
            probs.append(probs)
        losses.append(loss.item())

    test_loss = np.mean(losses)
    logger.success("\t End of testing. General loss: ", test_loss)
    wandb.log({"test_loss" : test_loss})
    return test_loss, lbls, probs #validation loss for a given epoch, labels and probs

def test_self_supervised(test_dataloader, model, loss_fn, configs, *metrics):
    '''
    Test `model` performance on data retrieved by `test_dataloader` in a self-supervised fashion.
    Performance are measure according to `loss_fn` and additional `metrics`.
    Params:
        - `test_dataloader`: instance of torch.utils.data.DataLoader to fetch data from test set
        - `model`: the model to test
        - `loss_fn`: the loss function which evaluate the model with
        - `metrics`: any additional metric which evaluate the model with
    Returns:
        - test_loss
    '''
    
    mask_token = torch.Tensor([([-1] * (configs['patch_width']//2)) + ([1] * (configs['patch_width']//2))] * configs['patch_height']).to(dtype=torch.float).cuda()


    patcher = Patcher((configs['patch_height'], configs['patch_width'])).cuda() 
    masker = Masker(mask_token, configs['mask_perc']).cuda() 

    
    model.eval()
    losses = []

    logger.info("\t Testing model performance...")
    
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        batch_ecg_data, batch_age, batch_sex = tuple(zip(*batch))

        batch_ecg_data = torch.stack(batch_ecg_data, dim=0).cuda().to(dtype=torch.float)
        batch_ecg_data, unfolded_shape = patcher(batch_ecg_data)

        if isinstance(loss_fn, PatchRecLoss):
            batch_masked_patches, batch_indeces_masked_patches = masker(batch_ecg_data.detach().clone())
            batch_age = torch.Tensor(batch_age).cuda().to(dtype=torch.float) # (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)
            with torch.no_grad():
                with autocast():
                    pred = model(batch_masked_patches, batch_age, batch_sex)
                    loss, rec_patches, real_patches = loss_fn(pred, batch_ecg_data, batch_indeces_masked_patches)
                    wandb.log({"Examples of reconstruction" : visual_comparing(rec_patches, real_patches, unfolded_shape, batch_indeces_masked_patches[0])}) #logging examples of reconstructed patches
            

        losses.append(loss.item())

    test_loss = np.mean(losses)

    logger.success("\t End of testing. General loss: ", test_loss)

    if isinstance(loss_fn, PatchRecLoss):
        wandb.log({"test_loss" : test_loss})
        return (test_loss, ) #test loss 