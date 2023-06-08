#Run.py

import wandb
import torch
from Model import FullModel
from torch.utils.data import DataLoader
from ECGDatasetSelfSupervised import ECGDatasetSelfSupervised
from ECGDatasetSupervised import ECGDatasetSupervised
from Configs import *
import matplotlib.pyplot as plt
import wandb
import logging
import os
from torch import nn
import numpy as np
from Optimizer import NoamOpt
from Training_and_testing import train_self_supervised, train_supervised, PatchRecLoss, test

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import sys

#NOTE: when running on 40gb gpu, max batch_size = 256
# hence, when running on 80gb gpu, max batch_size = 512

if __name__ == "__main__":


    # if not received any arguments, abort
    if len(sys.argv) < 2:
        print("Expected configuration file path as argument")
        exit(1)
    
    #if too many arguments, abort
    if len(sys.argv) > 2:
        print("Expected only one argument: configuration file path")
        exit(1)

    #check existence of configuration file path and is json
    if not os.path.exists(sys.argv[1]) and not sys.argv[1].endswith(".json"):
        print("Configuration file does not exist")
        exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available. CPU training not supported")
        exit(1)
    
# load configuration file
# NOTE: the configs.json file must be changed every time before a new run in order to test new configurations during training (also, machine capabilities must be taken into account)
configs = get_configs("./code/configs.json")

os.system("wandb login b73191cf01967a55aa21e93a37d3fc3ce6078859")


logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

#random choice between embedder_types options as well as decoding_type options and pos_enc_type options
configs['embedder_type'] = np.random.choice(configs['embedder_type'])
decoding_type = np.random.choice(configs['decoding_type'])
if decoding_type == 'None':
    configs['decoding_type'] = None
else:
    configs['decoding_type'] = decoding_type
configs['pos_enc_type'] = np.random.choice(configs['pos_enc_type'])

wandb.init(entity="cardi-ai", project="ECG-pretraining", config=configs, group="self-supervised")


#setting random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

#creating a dataset for self-supervised pre-training
train_set = ECGDatasetSelfSupervised("./train_self_supervised.csv", "./")
val_set = ECGDatasetSelfSupervised("./val_self_supervised.csv", "./")
test_set = ECGDatasetSelfSupervised("./test_self_supervised.csv", "./")

#creating a dataloader to generata batches
# no shuffle if distributed
train_dl = DataLoader(train_set, batch_size=configs['batch_size'], shuffle=(not distributed), collate_fn=lambda x: x, num_workers=configs["num_workers"],
                        pin_memory=pin_memory, sampler=(DistributedSampler(train_set) if distributed else None))
val_dl = DataLoader(val_set, batch_size=configs['batch_size'], shuffle=(not distributed), collate_fn=lambda x: x, num_workers=configs["num_workers"],
                        sampler = (DistributedSampler(val_set) if distributed else None), pin_memory=pin_memory)
test_dl = DataLoader(test_set, batch_size=configs['batch_size'], shuffle=True, collate_fn=lambda x: x, num_workers=configs["num_workers"],
                        pin_memory=pin_memory)

# Creation of model

import torch._dynamo
torch._dynamo.config.suppress_errors = True

model = FullModel(embedding_type=configs['embedder_type'],
                    patch_height=configs['patch_height'],
                    patch_width=configs['patch_width'],
                    n_patches=configs['n_patches'],
                    d_model=configs['d_model'], 
                    learnable_pos_enc=True,
                    p_dropout=.1, n_heads=configs['n_heads'],
                    dim_ff=configs['dim_ff'],
                    n_encoding_layers=configs['n_encoder_layers'], 
                    decoding_type=configs['decoding_type'],
                    pos_enc_type=configs['pos_enc_type'],
                    hiddem_dim=configs['hidden_dim'],
                    n_classes=configs['n_classes'])

# model = torch.compile(model)

if distributed: # use DDP
    device_ids = [0, 1]
    model = DDP(model, device_ids=device_ids)
else: # use single GPU
    print("Mapping model to cuda")
    model = model.cuda()

# print(model, "\n")

#xavier inatialization
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

#printing number of parameters
print(f"Number of parameters: {sum(p.data.nelement() for p in model.parameters())}")

#start training
optimizer = NoamOpt(configs['d_model'], 1, configs['warmup'], torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

wandb.watch(model, log="all") #logs all layers, gradients, parameters, etc. logs='all' logs histograms of parameters and gradients

#NOTE: model_name should be meaningful, something like the model architecture and the dataset used (e.g. lin_1d_enc_fcn, conv_2d_enc_rev, etc.)
training_loss, validation_loss = train_self_supervised(train_dl, model, optimizer, epochs=configs['epochs'], val_dataloader=val_dl,
                                                        early_stopping=True, patience=configs['patience'], save_model=True,
                                                        model_name="ss_test")

#visualizing results: plot training and validation loss over epochs
fig, _ = plt.plot(training_loss, label="Training loss")
fig, _ = plt.plot(validation_loss, label="Validation loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
fig.savefig("./loss_plot.png", bbox_inches='tight', dpi=300)

wandb.log({"loss_plot", fig})

#testing
test_loss = test(test_dl, model, PatchRecLoss())

wandb.log({'model' : model})

wandb.finish() #necesasry in notebooks