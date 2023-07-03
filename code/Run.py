#Run.py
import wandb
import torch
from Model import FullModel
from torch.utils.data import DataLoader
from ECGDatasetSelfSupervised import ECGDatasetSelfSupervised
from Configs import *
import matplotlib.pyplot as plt
import wandb
import logging
import os
from torch import nn
import numpy as np
from Optimizer import NoamOpt
from Train import train_self_supervised
from Test import test_self_supervised
from Losses import PatchRecLoss
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import sys
from loguru import logger
import datetime

def train():
    
    wandb.init(entity="cardi-ai", project="ECG-pretraining", config=wandb.config, group="self-supervised")
    
    #! careful when handling None decoding_type

    logger.info(f"Embedder type {wandb.config['embedder_type']}")
    logger.info(f"Positional encoding {wandb.config['pos_enc_type']}")
    logger.info(f"Decodying type {wandb.config['decoding_type']}")    

    #setting random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    logger.info("creating datasets for self-supervised pre-training...")

    train_set = ECGDatasetSelfSupervised("./train_self_supervised_processed.csv", "./train_self_supervised", wandb.config['reduced'])
    val_set = ECGDatasetSelfSupervised("./val_self_supervised_processed.csv", "./val_self_supervised", wandb.config['reduced'])
    test_set = ECGDatasetSelfSupervised("./test_self_supervised_processed.csv", "./test_self_supervised", wandb.config['reduced'])
    
    
    logger.info("creating dataloaders to generata batches...")

    # no shuffle if wandb.config["distributed"]
    train_dl = DataLoader(train_set, batch_size=wandb.config['batch_size'], shuffle=(not wandb.config["distributed"]),
                        collate_fn=lambda x: x, num_workers=wandb.config["num_workers"], sampler=(DistributedSampler(train_set) if wandb.config["distributed"] else None))
    val_dl = DataLoader(val_set, batch_size=wandb.config['batch_size'], shuffle=(not wandb.config["distributed"]),
                        collate_fn=lambda x: x, num_workers=wandb.config["num_workers"], sampler=(DistributedSampler(val_set) if wandb.config["distributed"] else None))
    test_dl = DataLoader(test_set, batch_size=wandb.config['batch_size'], shuffle=True, collate_fn=lambda x: x, num_workers=wandb.config["num_workers"])


    logger.info("creating model...")
    
    n_patches = (wandb.config['n_leads'] * wandb.config['window']) // (wandb.config['patch_height'] * wandb.config['patch_width'])

    model = FullModel(embedding_type=wandb.config['embedder_type'],
                        patch_height=wandb.config['patch_height'],
                        patch_width=wandb.config['patch_width'],
                        n_patches=n_patches,
                        d_model=wandb.config['d_model'], 
                        learnable_pos_enc=True,
                        p_dropout=.1, 
                        n_heads=wandb.config['n_heads'],
                        dim_ff=wandb.config['dim_ff'],
                        n_encoding_layers=wandb.config['n_encoder_layers'], 
                        decoding_type=wandb.config['decoding_type'],
                        pos_enc_type=wandb.config['pos_enc_type'],
                        hiddem_dim=wandb.config['hidden_dim'],
                        n_classes=wandb.config['n_classes'])

    # model = torch.compile(model)

    if wandb.config["distributed"]: # use DDP
        logger.info("Mapping model to distributed cuda")
        rank = dist.get_rank()
        world_size = torch.cuda.device_count()
        setup(rank, world_size)
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
    else: # use single GPU
        logger.info("Mapping model to cuda")
        model = model.cuda()

    # print(model, "\n")

    #xavier inatialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    #printing number of parameters
    logger.info(f"Number of parameters: {sum(p.data.nelement() for p in model.parameters())}")

    #start training
    optimizer = NoamOpt(wandb.config['d_model'], 1, wandb.config['warmup'], torch.optim.Adam(model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), eps=1e-9))


    #NOTE: model_name should be meaningful, something like the model architecture and the dataset used (e.g. lin_1d_enc_fcn, conv_2d_enc_rev, etc.)
    model_name = wandb.config['embedder_type'] + "_" + wandb.config['pos_enc_type'] + "_" + wandb.config['decoding_type']
    training_loss, validation_loss = train_self_supervised(train_dl,
                                                           model,
                                                           optimizer,
                                                           epochs=wandb.config['epochs'],
                                                           val_dataloader=val_dl,
                                                           configs=wandb.config,
                                                           early_stopping=wandb.config['early_stopping'],
                                                           patience=wandb.config['patience'],
                                                           save_model=True,
                                                           model_name=model_name)

    #visualizing results: plot training and validation loss over epochs
    # time_span = np.arange(0, len(training_loss))
    # fig = plt.plot(time_span, training_loss, label="Training loss")
    # fig = plt.plot(time_span, validation_loss, label="Validation loss")
    # plt.legend()
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")

    # wandb.log({"loss_plot", fig})

    #testing
    test_loss = test(test_dl, model, PatchRecLoss(), wandb.config)
    
    #logging model to wandb
    

    wandb.finish() #necesasry in notebooks
    
if __name__ == "__main__":
    # if not received any arguments, abort
    if len(sys.argv) < 2:
        logger.error("Expected yaml configuration file path as argument")
        exit(1)
    
    #if too many arguments, abort
    if len(sys.argv) > 2:
        logger.error("Expected only one argument: yaml configuration file path")
        exit(1)

    #check existence of configuration file path and is json
    if not os.path.exists(sys.argv[1]) and not sys.argv[1].endswith(".yaml"):
        logger.error("Configuration file does not exist or it is not a yaml file")
        exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA not available. CPU training not supported")
        exit(1)

#os.system("wandb login b73191cf01967a55aa21e93a37d3fc3ce6078859")
wandb.login()
   
# load configuration file
configs = get_configs_from_yaml(sys.argv[1])

#initialize sweep by passing wandb.config
sweep_id = wandb.sweep(
    sweep = configs,
    project = "ECG-pretraining"
)

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

wandb.agent(sweep_id, function=train, count=4)
