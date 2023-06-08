#Training_testing.py
import torch
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import wandb
import numpy as np
from Model import FullModel
from Configs import distributed
from torch.cuda.amp import autocast, GradScaler


global patience_count
global best_val_loss

class PatchRecLoss(nn.Module):
    def __init__(self, loss_type : str = 'mse'):
        super(PatchRecLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, batch_output, batch_real_patches, batch_indeces_masked_patches):
        # batch_output: (batch_size, n_patches, patch_height, patch_width)
        # batch_real_patches: (batch_size, n_patches, patch_height, patch_width)
        # batch_indeces_masked_patches: (batch_size, k), where k is the number of masked patches <= n_patches

        reconstructed_patches = []
        real_patches_to_compare = []

        batch_size, n_patches, patch_height, patch_width = batch_output.shape
        for i_instance in range(batch_size): #scanning instances in batch
            for i_patch in range(n_patches): #scanning patches in instance
                if i_patch in batch_indeces_masked_patches[i_instance]: #if patch was masked
                    reconstructed_patches.append(batch_output[i_instance, i_patch, :, :])
                    real_patches_to_compare.append(batch_real_patches[i_instance, i_patch, :, :])
        
        reconstructed_patches = torch.stack(reconstructed_patches, dim=0)
        real_patches_to_compare = torch.stack(real_patches_to_compare, dim=0)
        return nn.MSELoss()(reconstructed_patches, real_patches_to_compare), reconstructed_patches, real_patches_to_compare

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

def test(test_dataloader, model, loss_fn, *metrics):
    '''Testing of a model over a testing set. Returns test loss + additional performance depending on input metrics.'''

    model.eval()
    losses = []
    lbls = []
    probs = []
    sigmoid = nn.Sigmoid()

    example_patches = [] # for wandb reporting

    print("Testing model performance...")
    
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        batch_patches, batch_age, batch_sex, batch_labels = tuple(zip(*batch))

        #batch_age and batch_sex could be full on nan if self-supervised pretraining is used and instances in batch are taken from tnmg subsets
        # it's not a problem though because age and sex are not used in self-supervised pretraining 

        batch_patches = torch.stack(batch_patches, dim=0).cuda().to(dtype=torch.float)

        if isinstance(loss_fn, PatchRecLoss):
            batch_indeces_masked_patches, batch_real_patches = tuple(zip(*batch_labels)) #tuple(lists) and tuple(tensors)
            batch_real_patches = torch.stack(batch_real_patches, dim=0).cuda().to(dtype=torch.float) #all with shaped (n_patches, patch_height, patch_width) -> after stacking (batch_size, n_patches, patch_height, patch_width)
            batch_age = torch.Tensor(batch_age, ).cuda().to(dtype=torch.float) # normalized age, (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)
            with torch.no_grad():
                pred = model(batch_patches, batch_age, batch_sex)
                loss, rec_patches, real_patches = loss_fn(pred, batch_real_patches, batch_indeces_masked_patches)
                example_patches.append(wandb.Image(visual_comparing(rec_patches, real_patches)))
        else:
            #normalize age
            age_min = test_dataloader.dataset.ecg_dataframe['age'].min()
            age_max = test_dataloader.dataset.ecg_dataframe['age'].max()
            batch_age = torch.Tensor((batch_age - age_min)/(age_max - age_min)).cuda().to(dtype=torch.float) # normalized age, (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)
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

    print("End of testing. General loss: ", test_loss)

    if isinstance(loss_fn, PatchRecLoss):
        wandb.log({"test_loss" : test_loss, "Examples" : example_patches})
        return (test_loss, ) #test loss 
    else:
        wandb.log({"test_loss" : test_loss})
        return test_loss, lbls, probs #validation loss for a given epoch, labels and probs
    
def validate(val_dataloader, model, loss_fn, *metrics):
    '''Validation of a model over a validation set. Returns validation loss.'''
    model.eval()
    epoch_losses = []
    epoch_lbls = []
    epoch_probs = []
    sigmoid = nn.Sigmoid()
    
    for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

        batch_patches, batch_age, batch_sex, batch_labels = tuple(zip(*batch))

        #batch_age and batch_sex could be full on nan if self-supervised pretraining is used and instances in batch are taken from tnmg subsets
        # it's not a problem though because age and sex are not used in self-supervised pretraining 

        batch_patches = torch.stack(batch_patches, dim=0).cuda().to(dtype=torch.float)

        if isinstance(loss_fn, PatchRecLoss):
            batch_indeces_masked_patches, batch_real_patches = tuple(zip(*batch_labels)) #tuple(lists) and tuple(tensors)
            batch_real_patches = torch.stack(batch_real_patches, dim=0).cuda().to(dtype=torch.float).cuda #all with shaped (n_patches, patch_height, patch_width) -> after stacking (batch_size, n_patches, patch_height, patch_width)
            batch_age = torch.Tensor(batch_age).cuda().to(dtype=torch.float) # normalized age, (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)
            with torch.no_grad():
                with autocast():
                    pred = model(batch_patches, batch_age, batch_sex)
                    loss, rec_patches, real_patches = loss_fn(pred, batch_real_patches, batch_indeces_masked_patches)
        else:
            # normalize age
            age_min = val_dataloader.dataset.ecg_dataframe['age'].min()
            age_max = val_dataloader.dataset.ecg_dataframe['age'].max()
            batch_age = torch.Tensor((batch_age - age_min)/(age_max - age_min)).cuda().to(dtype=torch.float) # normalized age, (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)
            with torch.no_grad():
                with autocast():
                    batch_labels = torch.stack(batch_labels, dim=0).cuda().to(dtype=torch.float)
                    pred = model(batch_patches, batch_age, batch_sex)
                    loss = loss_fn(pred, batch_labels)
                    probs = sigmoid(pred).detach().cpu().numpy()
                    lbls = batch_labels.detach().cpu().numpy()
                    epoch_lbls.append(lbls)
                    epoch_probs.append(probs)

        epoch_losses.append(loss.item())

    val_loss = np.mean(epoch_losses)
    if isinstance(loss_fn, PatchRecLoss):
        wandb.log({"val_loss" : val_loss})
        return (np.mean(epoch_losses), ) #validation loss for a given epoch
    else:
        wandb.log({"val_loss" : val_loss})
        return val_loss, epoch_lbls, epoch_probs #validation loss for a given epoch, labels and probs for a given epoch

def save_model_modules(model, optimizer, path, model_name):
    if distributed:
        torch.save({
            'optimizer': optimizer.state_dict(),
            'model': model.module.state_dict(),
        }, path + model_name + "_full_model.pth")
    else:    
        torch.save({
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, path + model_name + "_full_model.pth")  

def train_supervised(train_datalaoder, model : FullModel, optimizer, epochs, val_dataloader, early_stopping=False, patience=5, save_model=False, model_name=None, *metrics):
    ''' Supervised training loop for a model given batches of instances generated by a train_datalaoder, an optimizer and the number of epochs.
    Returns training loss for every epoch + labels and corresponding output probs for every epoch.'''

    patience_count = 0
    best_val_loss = np.inf

    bce_loss_fn = nn.BCEWithLogitsLoss()
    training_loss = []
    validation_loss = []
    training_lbls_probs = []
    validation_lbls_probs = []

    model.train()

    for i in range(epochs):
        epoch_losses = []
        epoch_probs = []
        epoch_lbls = []
        print(f"Epoch {i+1}/{epochs}")
        if distributed:
            train_datalaoder.sampler.set_epoch(i)
        for j, batch in tqdm(enumerate(train_datalaoder), total=len(train_datalaoder)):

            batch_patches, batch_age, batch_sex, batch_labels = tuple(zip(*batch)) #return tuple of patches, tuple of ages, tuple of sexes, tuple of labels; each bs instances long       

            age_min = train_datalaoder.dataset.ecg_dataframe['age'].min()
            age_max = train_datalaoder.dataset.ecg_dataframe['age'].max()
            batch_age = torch.Tensor((batch_age - age_min)/(age_max - age_min)).cuda().to(dtype=torch.float) # normalized age, (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)

            #print(batch_age.shape, batch_sex.shape)

            batch_patches = torch.stack(batch_patches, dim=0).cuda().to(dtype=torch.float) 
            batch_labels = torch.stack(batch_labels, dim=0).cuda().to(dtype=torch.float)

            # compute prediction and loss
            prediction_logits = model(batch_patches, batch_age, batch_sex)
            loss = bce_loss_fn(prediction_logits, batch_labels)

            #backpropagation
            loss.backward()
            optimizer.step()
            optimizer.optimizer.zero_grad()

            epoch_losses.append(loss.item())
            epoch_probs.append(torch.sigmoid(prediction_logits).detach().cpu().numpy())
            epoch_lbls.append(batch_labels.detach().cpu().numpy())

        # end of epoch
        
        training_loss.append(np.mean(epoch_losses))
        training_lbls_probs.append((np.concatenate(epoch_lbls), np.concatenate(epoch_probs)))

        #validation
        if distributed:
            val_dataloader.sampler.set_epoch(i)
        epoch_val_loss, epoch_val_lbls, epoch_val_probs = validate(val_dataloader, model, bce_loss_fn)
        validation_loss.append(epoch_val_loss)
        validation_lbls_probs.append((np.concatenate(epoch_val_lbls), np.concatenate(epoch_val_probs)))

        print(f"Training loss: {training_loss[-1]} - Validation loss: {validation_loss[-1]}")

        #early stopping
        if early_stopping:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_count = 0
                if save_model and model_name is not None:
                    save_model_modules(model, optimizer, "./models/checkpoints/supervised/", model_name)
            else: #not improving
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {i+1}. Patience {patience} reached.")
                    break
    
    # end of training
    return training_loss, training_lbls_probs, validation_loss, validation_lbls_probs

def train_self_supervised(train_datalaoder, model, optimizer, epochs, val_dataloader, early_stopping=False, patience=5, save_model=False, model_name=None, *metrics):
    ''' Self-supervised training loop for a model given batches of instances generated by a train_datalaoder, an optimizer and the number of epochs.
    Returns training loss for every epoch.'''

    rec_loss_fn = PatchRecLoss(loss_type='mse')    
    training_loss = []
    validation_loss = []
    patience_count = 0
    best_val_loss = np.inf
    scaler = GradScaler()

    
    for i in range(epochs):
        epoch_losses = []
        print(f"Epoch {i+1}/{epochs}")
        if distributed:
            train_datalaoder.sampler.set_epoch(i)
        model.train()
        for j, batch in tqdm(enumerate(train_datalaoder), total=len(train_datalaoder)):

            optimizer.optimizer.zero_grad()

            batch_patches, batch_age, batch_sex, batch_labels = tuple(zip(*batch))

            #batch_age and batch_sex are full of nan. Don't care thouhg, they are not used in the self_superised training

            batch_age = torch.Tensor(batch_age).cuda().to(dtype=torch.float) # normalized age, (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)

            #print("batch age and sex shapes: ", batch_age.shape, batch_sex.shape) 

            
            
            batch_indeces_masked_patches, batch_real_patches = tuple(zip(*batch_labels)) #tuple(lists) and tuple(Tensors)
            
            batch_patches = torch.stack(batch_patches, dim=0).cuda().to(dtype=torch.float) # all shaped (n_patches, patch_height, patch_width) -> after stack (batch_size, n_patches, patch_height, patch_width)
            batch_real_patches = torch.stack(batch_real_patches, dim=0).cuda().to(dtype=torch.float) # all shaped (n_patches, patch_height, patch_width) -> after stack (batch_size, n_patches, patch_height, patch_width)         

            assert batch_patches.shape == batch_real_patches.shape

            #print("batch patches and real patches shapes: ", batch_patches.shape, batch_real_patches.shape)

            # compute prediction and loss
            with autocast():
                prediction = model(batch_patches, batch_age, batch_sex)
                loss, rec_patches, real_masked_patches = rec_loss_fn(prediction, batch_real_patches, batch_indeces_masked_patches)
            
            #backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer.optimizer)
            scaler.update()

            #append example loss
            epoch_losses.append(loss.item())

        # end of epoch
        training_loss.append(np.mean(epoch_losses))

        #validation
        if distributed:
            val_dataloader.sampler.set_epoch(i)
        epoch_val_loss = validate(val_dataloader, model, rec_loss_fn)[0]
        validation_loss.append(epoch_val_loss)

        print(f"Training loss: {training_loss[-1]} - Validation loss: {validation_loss[-1]}")

        wandb.log({"training_loss": training_loss[-1], "validation_loss": validation_loss[-1]})

        #early stopping
        if early_stopping:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_count = 0
                if save_model and model_name is not None:
                    save_model_modules(model, optimizer, "./models/checkpoints/self-supervised/", model_name)
            else: #not improving
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {i+1}, Patience {patience} reached.")
                    break

    #end of training
    wandb.log({"training_loss": training_loss, "validation_loss": validation_loss})
    return training_loss, validation_loss