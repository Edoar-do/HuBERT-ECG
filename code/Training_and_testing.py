#Training_testing.py
import torch
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import wandb
import numpy as np
from Model import FullModel, Patcher, Masker
from Configs import distributed, get_configs
from torch.cuda.amp import autocast, GradScaler
from Losses import PatchRecLoss

global configs 
global patience_count
global best_val_loss

global patcher
global masker


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
        - test_dataloader: instance of torch.utils.data.DataLoader to fetch data from test set
        - model: the model to test
        - loss_fn: the loss function which evaluate the model with
        - metrics: any additional metric which evaluate the model with
    Returns:
        - test_loss
        - test labels
        - test probabilities
    '''
    model.eval()
    losses = []
    lbls = []
    probs = []
    sigmoid = nn.Sigmoid()

    print("\t Testing model performance...")

    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_patches, batch_age, batch_sex, batch_labels = tuple(zip(*batch))
        batch_patches = torch.stack(batch_patches, dim=0).cuda().to(dtype=torch.float)
        batch_patches = patcher(batch_patches)
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
    print("\t End of testing. General loss: ", test_loss)
    wandb.log({"test_loss" : test_loss})
    return test_loss, lbls, probs #validation loss for a given epoch, labels and probs

def test_self_supervised(test_dataloader, model, loss_fn, *metrics):
    '''
    Test `model` performance on data retrieved by `test_dataloader` in a self-supervised fashion.
    Performance are measure according to `loss_fn` and additional `metrics`.
    Params:
        - test_dataloader: instance of torch.utils.data.DataLoader to fetch data from test set
        - model: the model to test
        - loss_fn: the loss function which evaluate the model with
        - metrics: any additional metric which evaluate the model with
    Returns:
        - test_loss
    '''

    model.eval()
    losses = []

    example_patches = [] # for wandb reporting

    print("\t Testing model performance...")
    
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        batch_patches, batch_age, batch_sex = tuple(zip(*batch))

        batch_patches = torch.stack(batch_patches, dim=0).cuda().to(dtype=torch.float)
        batch_patches = patcher(batch_patches)

        if isinstance(loss_fn, PatchRecLoss):
            batch_masked_patches, batch_indeces_masked_patches = masker(batch_patches.detach().clone())
            batch_age = torch.Tensor(batch_age).cuda().to(dtype=torch.float) # (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)
            with torch.no_grad():
                with autocast():
                    pred = model(batch_masked_patches, batch_age, batch_sex)
                    loss, rec_patches, real_patches = loss_fn(pred, batch_patches, batch_indeces_masked_patches)
            

        losses.append(loss.item())

    test_loss = np.mean(losses)

    print("\t End of testing. General loss: ", test_loss)

    if isinstance(loss_fn, PatchRecLoss):
        wandb.log({"test_loss" : test_loss, "Examples" : example_patches})
        return (test_loss, ) #test loss 


def validate_supervised(val_dataloader, model, loss_fn, *metrics):
    '''Validate a `model` in a supervised fashion once per epoch using data fetched by `val_dataloader` according to `loss_fn` function and any additional `metric`.
    Params:
        - val_dataloader: instance of torch.utils.data.DataLoader that fetches data from validation set
        - model: the model to validate
        - loss_fn: the loss function to validate the model on
        - metrics: any additional metrics to validate the model on
    Returns:
        - validation loss for the epoch
        - [labels, probabilities] if supervised trainining (opt.)
    '''
    model.eval()
    epoch_losses = []
    epoch_lbls = []
    epoch_probs = []
    sigmoid = nn.Sigmoid() 

    print("\t Validating model performace...")

    for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

        batch_ecg_data, batch_age, batch_sex, batch_labels = tuple(zip(*batch))

        batch_ecg_data = torch.stack(batch_ecg_data, dim=0).cuda().to(dtype=torch.float)
        batch_patches = patcher(batch_ecg_data)

        # normalize age
        age_min = val_dataloader.dataset.ecg_dataframe['age'].min()
        age_max = val_dataloader.dataset.ecg_dataframe['age'].max()
        batch_age = torch.Tensor((batch_age - age_min)/(age_max - age_min)).cuda().to(dtype=torch.float) # normalized age, (batch_size, 1)

        batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)

        batch_labels = torch.stack(batch_labels, dim=0).cuda().to(dtype=torch.float)

        with torch.no_grad():
            with autocast():
                pred = model(batch_patches, batch_age, batch_sex)
                loss = loss_fn(pred, batch_labels)
                probs = sigmoid(pred).detach().cpu().numpy()
                lbls = batch_labels.detach().cpu().numpy()
                epoch_lbls.append(lbls)
                epoch_probs.append(probs) 
        
        epoch_losses.append(loss.item())

    val_loss = np.mean(epoch_losses)
    wandb.log({"val_loss" : val_loss})
    return val_loss, epoch_lbls, epoch_probs #validation loss for a given epoch, labels and probs for a given epoch


def validate_self_supervised(val_dataloader, model, loss_fn, *metrics):
    '''Validate a `model` in a self_supervised fashion once per epoch using data fetched by `val_dataloader` according to `loss_fn` function and any additional `metric`.
    Params:
        - val_dataloader: instance of torch.utils.data.DataLoader that fetches data from validation set
        - model: the model to validate
        - loss_fn: the loss function to validate the model on
        - metrics: any additional metrics to validate the model on
    Returns:
        - validation loss for the epoch
        - [labels, probabilities] if supervised trainining (opt.)
    '''
    model.eval()
    epoch_losses = []

    print("\t Validating model performance...")
    
    for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

        batch_ecg_data, batch_age, batch_sex = tuple(zip(*batch))

        #batch_ecg_data is a tuple of torch.Tensor (12, 5000) that is batch_size long
        #batch_age, batch_sex are tuples on nan (float) that are batch_size long in case of self-supervised (supervised) training

        batch_ecg_data = torch.stack(batch_ecg_data, dim=0).cuda().to(dtype=torch.float) #--> (bs, 12, 5000)

        batch_patches = patcher (batch_ecg_data)
        

        if isinstance(loss_fn, PatchRecLoss):
            batch_masked_patches, batch_indeces_masked_patches = masker(batch_patches.detach().clone())
            batch_age = torch.Tensor(batch_age).cuda().to(dtype=torch.float) # (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)
            with torch.no_grad():
                with autocast():
                    pred = model(batch_masked_patches, batch_age, batch_sex)
                    loss, rec_patches, real_patches = loss_fn(pred, batch_patches, batch_indeces_masked_patches)

        epoch_losses.append(loss.item())

    val_loss = np.mean(epoch_losses)
    if isinstance(loss_fn, PatchRecLoss):
        wandb.log({"val_loss" : val_loss})
        return np.mean(epoch_losses) #validation loss for a given epoch

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

#TODO: da rivedere train_supervised
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
    ''' Train a `model` passed as parameter in a self-supervised fashion for `epochs` epochs, using data provided by a `train_dataloader`.
    The evaluation throughout training is done using data provided by a `val_dataloader` and can take into account additional `metrics`.
    Params:
        - train_dataloader: instance of torch.utils.data.DataLoader that retrives batches of data from training set
        - model: the model to train and optimize
        - optimizer: the optimizer to be used
        - epochs: integer number of epochs to train the model
        - val_dataloader: instance of torch.utils.data.DataLoader that retrieves batches of data from validation set
        - early_stopping: bool parameter to indicate whether to apply earlt stopping condition
        - patience: integer number of epochs within which no improvement on validation set is tolerated
        - save_model: bool param to indicate whether to save the model
        - model_name: str to indicate the model name
        - *metrics: any additional metric to evaluate the model on (not implemented)
    Returns:
        - training loss (list[float])
        - validation loss (list[float])
    '''

    configs = get_configs("./ECG_pretraining/code/configs.json") #global var initialized here, once and forever

    rec_loss_fn = PatchRecLoss(loss_type='mse')    
    training_loss = []
    validation_loss = []
    patience_count = 0 #global var initialized here, once and forever
    best_val_loss = np.inf #global var initialized here, once and forever
    scaler = GradScaler()

    patcher = Patcher((configs['patch_height'], configs['patch_width'])).cuda() #global var initialized here, once and forever
    masker = Masker(configs['mask_token'], configs['mask_perc']).cuda() #global var initialized here, once and forever

    
    for i in range(epochs):
        epoch_losses = []
        print(f"Epoch {i+1}/{epochs}")
        if distributed:
            train_datalaoder.sampler.set_epoch(i)
        model.train()
        for j, batch in tqdm(enumerate(train_datalaoder), total=len(train_datalaoder)):

            optimizer.optimizer.zero_grad()

            batch_ecg_data, batch_age, batch_sex = tuple(zip(*batch))

            #batch_age and batch_sex are tuples of nan. Don't care though, they are not used in the self_supervised training

            batch_age = torch.Tensor(batch_age).cuda().to(dtype=torch.float) # (batch_size, 1)
            batch_sex = torch.Tensor(batch_sex).cuda().to(dtype=torch.float) # (batch_size, 1)

            # batch_ecg_data is a tuple of torch.Tensor (12, 5000) batch_size long
            batch_ecg_data = torch.stack(batch_ecg_data, dim=0).cuda().to(dtype=torch.float)
            #now it's a torch.Tensor (batch_size, 12, 5000)

            batch_patches = patcher (batch_ecg_data) # (batch_size, n_patches, patch_height, patch_width)
            batch_masked_patches, batch_indeces_masked_patches = masker(batch_patches.detach().clone())
            
            #all the above tensors should be on cuda since all deriva from batch_ecg_data that is on cuda

            assert batch_patches.shape == batch_masked_patches.shape
            assert batch_patches.shape[0] == len(batch_indeces_masked_patches)

            # compute prediction and loss
            with autocast():
                prediction = model(batch_masked_patches, batch_age, batch_sex)
                loss, rec_patches, real_masked_patches = rec_loss_fn(prediction, batch_patches, batch_indeces_masked_patches)
            
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
        epoch_val_loss = validate_self_supervised(val_dataloader, model, rec_loss_fn)
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