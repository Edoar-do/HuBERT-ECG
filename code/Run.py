#Run.py
import wandb
import torch
from Model import FullModel
from torch.utils.data import DataLoader
from ECGDatasetSelfSupervised import ECGDatasetSelfSupervised
from Configs import *
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
import sys
from loguru import logger
import json
import datetime

def train():
    
    wandb.init(entity="cardi-ai", project="ECG-pretraining", config=wandb.config, group="self-supervised")
    
    # logger.info(f"{wandb.config}")
    
    #! careful when handling None decoding_type (only when supervised training)

    logger.info(f"Embedder type {wandb.config['embedding_type']}")
    logger.info(f"Positional encoding {wandb.config['pos_enc_type']}")
    logger.info(f"Decodying type {wandb.config['decoding_type']}")    

    #setting random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    logger.info("creating datasets for self-supervised pre-training...")

    train_set = ECGDatasetSelfSupervised("../../train_self_supervised_processed.csv", "../../train_self_supervised", wandb.config['reduced'])
    val_set = ECGDatasetSelfSupervised("../../val_self_supervised_processed.csv", "../../val_self_supervised", wandb.config['reduced'])
    test_set = ECGDatasetSelfSupervised("../../test_self_supervised_processed.csv", "../../test_self_supervised", wandb.config['reduced'])
  
    
    logger.info("creating dataloaders to generata batches...")

    # no shuffle if distributed
    train_dl = DataLoader(train_set, batch_size=wandb.config['batch_size'], shuffle=(not wandb.config['distributed']),
                        collate_fn=lambda x: x, num_workers=4, sampler=(DistributedSampler(train_set) if wandb.config['distributed'] else None))
    val_dl = DataLoader(val_set, batch_size=wandb.config['batch_size'], shuffle=(not wandb.config['distributed']),
                        collate_fn=lambda x: x, num_workers=4, sampler=(DistributedSampler(val_set) if wandb.config['distributed'] else None))
    test_dl = DataLoader(test_set, batch_size=wandb.config['batch_size'], shuffle=True, collate_fn=lambda x: x, num_workers=4)


    logger.info("creating model...")
    
    n_patches = (wandb.config['n_leads'] * wandb.config['window']) // (wandb.config['patch_height'] * wandb.config['patch_width'])

    model = FullModel(embedding_type=wandb.config['embedding_type'],
                        patch_height=wandb.config['patch_height'],
                        patch_width=wandb.config['patch_width'],
                        n_patches=n_patches,
                        d_model=wandb.config['d_model'], 
                        learnable_pos_enc=True,
                        p_dropout=.1, 
                        n_heads=wandb.config['n_heads'],
                        dim_ff=wandb.config['d_model'] * wandb.config['dim_ff_ratio'],
                        n_encoding_layers=wandb.config['n_encoding_layers'],  
                        decoding_type=wandb.config['decoding_type'],
                        pos_enc_type=wandb.config['pos_enc_type'],
                        hiddem_dim=None,
                        n_classes=wandb.config['n_classes'],
                        dim_ff_decoding = wandb.config['d_model'] * wandb.config['dim_ff_ratio'],
                        n_decoding_heads = wandb.config['n_heads'],
                        n_decoding_layers = wandb.config['n_encoding_layers'])

    # model = torch.compile(model)

    if wandb.config['distributed']: # use DDP
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

    model_name = wandb.config['embedding_type'] + "_" + wandb.config['pos_enc_type'] + "_" + wandb.config['decoding_type'] + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

    #testing
    test_loss = test_self_supervised(test_dl, model, PatchRecLoss(), wandb.config)
    

    #wandb.finish() #necesasry in notebooks
    
if __name__ == "__main__":
    
    #the following checks are to be done only if all sweeps and agents are created in the code and need params from shell {  
    # if not received any arguments, abort
    # if len(sys.argv) < 2:
    #     logger.error("Expected yaml configuration file path as argument, and optionally a sweep id")
    #     exit(1)
    
    # #if too many arguments, abort
    # if len(sys.argv) > 3:
    #     logger.error("Expected at most two arguments: yaml configuration file path and, optionally, a sweep id")
    #     exit(1)
    # #check existence of configuration file path and yaml extension
    # if not os.path.exists(sys.argv[1]) and not sys.argv[1].endswith(".yaml"):
    #     logger.error("Configuration file does not exist or it is not a yaml file")
    #     exit(1)

    # if not torch.cuda.is_available():
    #     logger.error("CUDA not available. CPU training not supported")
    #     exit(1)
    
    # }

    # #os.system("wandb login b73191cf01967a55aa21e93a37d3fc3ce6078859")
    # wandb.login()
    
    train()
   
    # the following section has to be inserted only if sweep and agents are used from script. If from CLI not necessary{
        
    # # load configuration file
    # configs = get_configs_from_yaml(sys.argv[1])
    # # configs = get_configs(sys.argv[1])
    

    # # attach to an already intialize sweep using the sweep id or initialize sweep by passing configs
    # if len(sys.argv) == 3 and sys.argv[2] is not None:
    #     sweep_id = str(sys.argv[2])
    # else:
    #     sweep_id = wandb.sweep(
    #         sweep = configs,
    #         project = "ECG-pretraining"
    #     )
        

    # # logging.propagate = False
    # # logging.getLogger().setLevel(logging.ERROR)
    
    # #train(configs)

    # wandb.agent(sweep_id, function=train, count=10)
    
    # }
    

