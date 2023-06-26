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


def visual_comparing(rec_patches, real_patches):
    '''Returns a figure with n_patches subplots, each containing a pair of reconstructed and real patches.'''
    #rec_patches and real_patches: (batch_size, n_patches, patch_height, patch_width)
    #compare only first instance in batch and maximum 5 patches
    batch_size, n_patches, patch_height, patch_width = rec_patches.shape
    fig, axs = plt.subplots(n_patches, patch_height, figsize=(20, 20))
    for i_patch in range(max(5, n_patches)):
        for lead in range(patch_height):
            axs[i_patch, lead].plot(rec_patches[0, i_patch, lead, :].detach().cpu().numpy(), label='reconstructed')
            axs[i_patch, lead].plot(real_patches[0, i_patch, lead, :].detach().cpu().numpy(), label='real')
    
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
        batch_patches, batch_age, batch_sex, batch_labels = tuple(zip(*batch))
        batch_patches = torch.stack(batch_patches, dim=0).cuda().to(dtype=torch.float)
        batch_patches = patcher(batch_patches)
        #normalize age

        batch_age = torch.Tensor((batch_age - age_min)/(age_max - age_min), dtype=torch.float).cuda() # normalized age, (batch_size, 1)
        batch_sex = torch.Tensor(batch_sex, dtype=torch.float).cuda()# (batch_size, 1)
        with torch.no_grad():
            batch_labels = torch.stack(batch_labels, dim=0).cuda().to(dtype=torch.float)
            pred = model(batch_patches, batch_age, batch_sex)
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

def test_self_supervised(test_dataloader, model, loss_fn, *metrics):
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
    
    configs = get_configs("./ECG_pretraining/code/configs.json")

    patcher = Patcher((configs['patch_height'], configs['patch_width'])).cuda() 
    masker = Masker(configs['mask_token'], configs['mask_perc']).cuda() 

    
    model.eval()
    losses = []

    example_patches = [] # for wandb reporting

    logger.info("\t Testing model performance...")
    
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        batch_patches, batch_age, batch_sex = tuple(zip(*batch))

        batch_patches = torch.stack(batch_patches, dim=0).cuda().to(dtype=torch.float)
        batch_patches = patcher(batch_patches)

        if isinstance(loss_fn, PatchRecLoss):
            batch_masked_patches, batch_indeces_masked_patches = masker(batch_patches.detach().clone())
            batch_age = torch.Tensor(batch_age, dtype=torch.float).cuda() # (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex, dtype=torch.float).cuda() # (batch_size, 1)
            with torch.no_grad():
                with autocast():
                    pred = model(batch_masked_patches, batch_age, batch_sex)
                    loss, rec_patches, real_patches = loss_fn(pred, batch_patches, batch_indeces_masked_patches)
                    example_patches.append(wandb.Image(visual_comparing(rec_patches, real_patches)))
            

        losses.append(loss.item())

    test_loss = np.mean(losses)

    logger.success("\t End of testing. General loss: ", test_loss)

    if isinstance(loss_fn, PatchRecLoss):
        wandb.log({"test_loss" : test_loss, "Examples" : example_patches})
        return (test_loss, ) #test loss 