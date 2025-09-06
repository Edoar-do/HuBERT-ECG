import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from loguru import logger
import argparse
from tqdm import tqdm
from hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import wandb
import numpy as np
import copy
from math import ceil
from dataset import ECGDataset
import os
import random
from transformers.models.hubert.modeling_hubert import compute_mask_indices
from transformers import HubertConfig

EPS = 1E-09
MINIMAL_IMPROVEMENT = 1e-3
DROPOUT_DYNAMIC_REG_FACTOR = 0.05

SELF_SUPERVISED_MODEL_CKPT_PATH = "/path/to/models/checkpoints/self-supervised/"

def dynamic_regularizer(optimizer, model, penalty):
    if penalty:
        # penalizing model with regularization but not too much
        optimizer.param_groups[0]['weight_decay'] *= 5
        for name, module in model.named_modules():
            if 'dropout' in name:
                module.p += DROPOUT_DYNAMIC_REG_FACTOR
    else:
        # unburdening model from regularization
        # minimum attainable weight decay is 0.01, dropout is 0.1
        optimizer.param_groups[0]['weight_decay'] = max(0.01, optimizer.param_groups[0]['weight_decay'] / 5)
        for name, module in model.named_modules():
            if 'dropout' in name:
                module.p = max(0.1, module.p - DROPOUT_DYNAMIC_REG_FACTOR)
        

def train(args):
     
    device = torch.device('cuda')
    
    ### NOTE: comment for sweeps, uncomment for normal run ###
    wandb.init(entity="my-entity", project="my-project", group="self-supervised")

    if args.wandb_run_name is not None:
        wandb.run.name = args.wandb_run_name

    ### fixing seeds ###
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    ### configs ###
    patience = args.patience if args.patience is not None else args.training_steps // args.val_interval
    lr = args.lr
    betas = (0.9, 0.98)
    weight_decay = max(0, 0.01 * args.weight_decay_mult)
    accumulation_steps = args.accumulation_steps
    mask_time_prob = args.mask_time_prob
    
    ### size model hyperparams ###
    if args.largeness == "base":
        hidden_size = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        intermediate_size = 3072    
        classifier_proj_size = 256
        layerdrop = 0.1
    elif args.largeness == "large":
        hidden_size = 960
        num_hidden_layers = 16
        num_attention_heads = 12
        intermediate_size = 3840
        classifier_proj_size = 512
        layerdrop = 0.0
    elif args.largeness == 'small': # small
        hidden_size = 512
        num_hidden_layers = 8
        num_attention_heads = 8
        intermediate_size = 2048
        classifier_proj_size = 256
        layerdrop = 0.1
    else:
        raise ValueError(f"Model largeness {args.largeness} not supported")
    
        
    if args.resume_pretraining:
        hubert_name = args.load_path.split('/')[-1]
        logger.info(f"Loading checkpoint {hubert_name} to resume pretraining")
        
        checkpoint = torch.load(args.load_path, map_location = torch.device('cpu'))

        config = checkpoint['model_config']
        assert checkpoint['pretraining_vocab_sizes'] == args.vocab_sizes
        if type(config) == HubertConfig:
            config = HuBERTECGConfig(ensemble=len(checkpoint['pretraining_vocab_sizes']), vocab_sizes=checkpoint['pretraining_vocab_sizes'], **config.to_dict())
       
        hubert = HuBERT(config)
        hubert.load_state_dict(checkpoint['model_state_dict'])

        previous_iteration = int(hubert_name.split('_')[1])

        if args.train_iteration != previous_iteration: #when switching to subsequent training iterations
            logger.info("Switching to another pretraining iteration: changing label embedding and restoring dropouts...")
            hubert.label_embedding = nn.ModuleList(nn.Embedding(vocab_size, hubert.config.classifier_proj_size) for vocab_size in args.vocab_sizes)
            
            for name, module in hubert.named_modules():
                if 'dropout' in name and 'encoder.layers' in name:
                    module.p = 0.1 # restoring p drop
                    
        # hubert = nn.DataParallel(hubert)
        hubert.to(device)
        global_step = checkpoint['global_step'] if args.train_iteration == previous_iteration else 0 
        best_val_loss = checkpoint['best_val_loss'] if args.train_iteration == previous_iteration else float('inf')
        patience_count = checkpoint['patience_count'] if args.train_iteration == previous_iteration else 0
        best_val_accuracy = checkpoint['best_val_accuracy'] if args.train_iteration == previous_iteration else 0
        
        optimizer = optim.AdamW(
            hubert.parameters(),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        
        if args.train_iteration == previous_iteration: #don't load state dict when switching to subsequent train iterations
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.param_groups[0]['weight_decay'] = max(0.01, optimizer.param_groups[0]['weight_decay'])
            for name, module in hubert.named_modules():
                if 'dropout' in name:
                    module.p = max(0.1, module.p)
        
        if args.train_iteration == previous_iteration:
            lr_scheduler = get_linear_schedule_with_warmup(
               optimizer=optimizer,
               num_warmup_steps=ceil(0.08*args.training_steps  - global_step),
               num_training_steps=args.training_steps,
               last_epoch=checkpoint['lr_scheduler_state_dict']['last_epoch']-1
            )
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=ceil(0.08*args.training_steps),
                num_training_steps=args.training_steps
            )
        
        logger.info("Checkpoint loaded.")
    else:
        logger.info("Building a model from zero to start training...")
        
        if args.downsampling_factor is None:
            conv_kernel = (10, 3, 3, 3, 3, 2, 2)
            conv_stride = (5, 2, 2, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512, 512, 512)   
        elif args.downsampling_factor == 5: 
            conv_kernel = (10, 3, 3, 2, 2)
            conv_stride = (4, 2, 2, 2, 2)
            conv_dim = (512, 512, 512, 512, 512)
        elif args.downsampling_factor == 10:
            conv_kernel = (10, 3, 3, 2)
            conv_stride = (4, 2, 2, 2)
            conv_dim = (512, 512, 512, 512)
        else:
            raise ValueError(f"Downsampling factor {args.downsampling_factor} not supported")           
            
                
        config = HuBERTECGConfig(
            ensemble_length=len(args.vocab_sizes),
            vocab_sizes=args.vocab_sizes,
            hidden_size = hidden_size,
            num_hidden_layers = num_hidden_layers,
            num_attention_heads = num_attention_heads,
            intermediate_size = intermediate_size,
            mask_time_prob = mask_time_prob, 
            classifier_proj_size = classifier_proj_size,
            layerdrop = layerdrop,
            conv_kernel = conv_kernel,
            conv_stride = conv_stride,
            conv_dim = conv_dim,
            mask_time_length = 1,
            hidden_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            activation_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            attention_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            feat_proj_dropout=max(0, 0 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),
            final_dropout=max(0, 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * args.model_dropout_mult),    
        ) # + other default params
        
        hubert = HuBERT(config)
        # hubert = nn.DataParallel(hubert)
        hubert.to(device)
        global_step = 0
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        patience_count = 0        
        optimizer = optim.AdamW(
            hubert.parameters(),
            lr=lr,
            betas=betas,
            eps=EPS,
            weight_decay=weight_decay,
        )
        logger.info("Model built.")
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=ceil(0.08*args.training_steps), num_training_steps=args.training_steps)

    scaler = amp.GradScaler()
    
    # number of params
    logger.info(f"Number of parameters: {sum(p.numel() for p in hubert.parameters())}")
    
        
    ### START TRAINING ITERATION ###
    
    train_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_train,
        ecg_dir_path="/data/ECG_AF/train_self_supervised",
        downsampling_factor = args.downsampling_factor,
        features_path=args.train_features_path,
        kmeans_path = args.kmeans_path,
        )

    val_set = ECGDataset(
        path_to_dataset_csv=args.path_to_dataset_csv_val,
        ecg_dir_path="/data/ECG_AF/val_self_supervised",
        features_path=args.val_features_path,
        downsampling_factor = args.downsampling_factor,
        kmeans_path = args.kmeans_path,
        )
    
    assert len(args.vocab_sizes) == train_set.ensamble_length, f"len(vocab_sizes) must be equal to the number of tasks. Found {len(args.vocab_sizes)} and {train_set.ensamble_length} tasks"
    for v, k in zip(args.vocab_sizes, train_set.ensamble_kmeans):
        assert v == k.cluster_centers_.shape[0], f"vocab_sizes must be equal to the number of clusters in the kmeans models. Found {v} and {k.cluster_centers_.shape[0]} clusters"
        
    
    train_dl = DataLoader(
        train_set,
        collate_fn=train_set.collate,
        num_workers=6,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
        )

    val_dl = DataLoader(
        val_set,
        collate_fn=val_set.collate,
        num_workers=6,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
        )

    epochs = args.training_steps // (len(train_dl) // accumulation_steps) + 1 if args.training_steps is not None else args.epochs

    start_epoch = global_step // len(train_dl)
            
    for epoch in range(start_epoch, epochs):

        hubert.train()
        logger.info(f"Epoch {epoch+1}/{epochs}")

        train_losses = []
        
        for ecg, attention_mask, ensamble_labels in tqdm(train_dl, total=len(train_dl)):

            global_step += 1
            
            ecg = ecg.to(device) 
            attention_mask = attention_mask.to(device)
            ensamble_labels = ensamble_labels.to(device)
            
            #logger.info("Mapped data to device")

            with amp.autocast():
               
                out_encoder_dict = hubert(ecg, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=True)
                #logger.info("Computed encodings")

                mask = out_encoder_dict['mask_time_indices']
                
                ensamble_logits = hubert.logits(out_encoder_dict['last_hidden_state'])
                #logger.info("Computed logits")
                                
                # modify loss computation to enable ensamble loss (sum of losses)                
                ensamble_labels = ensamble_labels.transpose(0, 1) 
                
                masked_loss = 0
                unmasked_loss = 0
                
                assert len(ensamble_labels) == len(ensamble_logits), f"len(ensamble_labels) must be equal to len(ensamble_logits). Found {len(ensamble_labels)} and {len(ensamble_logits)}"
                
                for labels, logits in zip(ensamble_labels, ensamble_logits):
                    # labels is (BS, F), logits is (BS, F, V)
                    masked_loss += F.cross_entropy(logits[mask], labels[mask])
                    unmasked_loss += F.cross_entropy(logits[~mask], labels[~mask])
                    #logger.info("Computed masked and unmasked losses per task")
                    
                loss = args.alpha * masked_loss +  (1 - args.alpha) * unmasked_loss
                loss = loss / accumulation_steps
                       
            scaler.scale(loss).backward()
            train_losses.append(loss.item())
            
            #logger.info("Accumulated scaled loss")
            
            ### GRADIENT ACCUMULATION ###
            
            if global_step % accumulation_steps == 0:
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()                

            ### VALIDATION LOOP EVERY `val_interval` STEPS + LOGGING + CHECK OF EARLY STOPPING CONDITION ###

            if global_step % args.val_interval == 0:

                hubert.eval()
                
                val_losses = []                
                val_accuracies = []
                
                logger.info(f"Validating model at step {global_step}...")
                
                ### VALIDATION LOOP ###
                
                for ecg, _, ensamble_labels in tqdm(val_dl, total=len(val_dl)):
                    ecg = (ecg).to(device)
                    #attention_mask = (attention_mask).to(device) # attention mask could harm inference performance according to HF docs
                    ensamble_labels = (ensamble_labels).to(device)
                    
                    ensamble_labels = ensamble_labels.transpose(0, 1) # (ensamble_length, BS, F)

                    with torch.no_grad():
                        out_encoder_dict = hubert(ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
                        ensamble_logits = hubert.logits(out_encoder_dict['last_hidden_state'])
                        
                        assert len(ensamble_labels) == len(ensamble_logits), f"VAL! len(ensamble_labels) must be equal to len(ensamble_logits). Found {len(ensamble_labels)} and {len(ensamble_logits)}"
                        
                        loss = 0
                        accuracy = 0
                        for labels, logits in zip(ensamble_labels, ensamble_logits):
                            logits = logits.transpose(1, 2)
                            loss += F.cross_entropy(logits, labels)
                            accuracy += (logits.argmax(dim=1) == labels).float().mean() # mean over batch for a given task
                        
                        accuracy /= len(ensamble_logits) # mean over tasks
                        
                    val_accuracies.append(accuracy.item())                    
                    val_losses.append(loss.item())
                    
                ### END OF VALIDATION LOOP ###
                    
                val_loss = np.mean(val_losses)
                val_accuracy = np.mean(val_accuracies)
                train_loss = np.mean(train_losses)
                train_losses.clear() # to keep it aligned with validation losses
                    
                ### LOGGING ###
                
                logger.info(f"Step: {global_step}")
                logger.info(f"train_loss_{args.train_iteration}: {train_loss}")
                logger.info(f"val_loss_{args.train_iteration}: {val_loss}")
                logger.info(f"val_accuracy: {val_accuracy}")
                
                                                                        
                wandb.log({
                    f"train_loss_{args.train_iteration}" : train_loss,
                    f"val_loss_{args.train_iteration}" : val_loss,
                    "val_accuracy" : val_accuracy
                })

                hubert.train()

                ### SAVE IF NEW BEST MODEL + EARLY STOPPING ###
                if val_loss <= best_val_loss - MINIMAL_IMPROVEMENT: # if loss improves significantly, save checkpoint
                    
                    best_val_loss = val_loss
                    best_val_accuracy = val_accuracy if val_accuracy > best_val_accuracy else best_val_accuracy
                    patience_count = 0 
                    checkpoint = {
                                    "global_step" : global_step,
                                    "patience_count" : patience_count,
                                    "model_config" : hubert.config,
                                    "model_state_dict" : copy.deepcopy(hubert.state_dict()),
                                    "optimizer_state_dict" : copy.deepcopy(optimizer.state_dict()),
                                    "best_val_loss" : best_val_loss,
                                    "lr_scheduler_state_dict" : copy.deepcopy(lr_scheduler.state_dict()),
                                    "best_val_accuracy" : best_val_accuracy,
                                    "pretraining_vocab_sizes" : args.vocab_sizes,
                                }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_{wandb.run.id}.pt"
                    torch.save(checkpoint, SELF_SUPERVISED_MODEL_CKPT_PATH + checkpoint_name )

                    logger.info(f"New best (best_val_loss = {best_val_loss}) - model saved at step {global_step}")
                    
                    dynamic_regularizer(optimizer, hubert, penalty=False) if args.dynamic_reg else None # unburdening model from regularization

                elif val_accuracy >= best_val_accuracy + MINIMAL_IMPROVEMENT: # if loss doesn't improve significantly but accuracy does, save checkpoint anyway
                    
                    best_val_accuracy = val_accuracy
                    checkpoint = {
                                    "global_step" : global_step,
                                    "patience_count" : patience_count,
                                    "model_config" : hubert.config,
                                    "model_state_dict" : copy.deepcopy(hubert.state_dict()),
                                    "optimizer_state_dict" : copy.deepcopy(optimizer.state_dict()),
                                    "best_val_loss" : best_val_loss,
                                    "lr_scheduler_state_dict" : copy.deepcopy(lr_scheduler.state_dict()),
                                    "best_val_accuracy" : best_val_accuracy,
                                    "pretraining_vocab_sizes" : args.vocab_sizes,
                                }
                    
                    checkpoint_name = f"hubert_{args.train_iteration}_iteration_{global_step}_{wandb.run.id}.pt"                                          
                    
                    torch.save(checkpoint,  SELF_SUPERVISED_MODEL_CKPT_PATH + checkpoint_name)
                    logger.info(f"Val loss not improved but val accuracy did (best_val_accuracy = {best_val_accuracy}) - model saved at step {global_step}")   
                    
                    dynamic_regularizer(optimizer, hubert, penalty=False) if args.dynamic_reg else None # unburdening model from regularization
                    
                else: #worsening performance
                    patience_count += 1
                     
                    if args.dynamic_reg and patience_count % (patience // args.intervals_for_penalty) == 0 and patience_count != patience:
                        dynamic_regularizer(optimizer, hubert, penalty=True) # penalizing model with regularization
                    
                    if patience_count == patience:
                        logger.warning(f"EARLY STOPPING: Max num of val intervals with no improvement reached at {global_step}")
                        wandb.log({
                            "patience_count" : patience_count
                        })
                        return
                    

    ### END OF TRAINING ITERATION ###
    logger.info("End of training")
    logger.info(f"STATS: Global step={global_step}, Best val loss={best_val_loss}")
    wandb.finish()


if __name__ == "__main__":
    
    ### NOTE: comment for sweeps, uncomment for normal run ###

    parser = argparse.ArgumentParser(description="Train Hubert-ECG")

    #train_iteration
    parser.add_argument(
        "train_iteration",
        help="Hubert training iteration in {1, 2, 3}", 
        type=int,
        choices=[1, 2, 3]
    )
    
    #path_to_dataset_csv_train
    parser.add_argument(
        "path_to_dataset_csv_train",
        help="Path to the csv file containing the training dataset",
        type=str
    )
    
    #path_to_dataset_csv_val
    parser.add_argument(
        "path_to_dataset_csv_val",
        help="Path to the csv file containing the validation dataset",
        type=str
    )
    
    #training_steps
    parser.add_argument(
        "--training_steps",
        help="[OPT.] Number of training steps to perform",
        type=int,
    )
    
    #epochs
    parser.add_argument(
        "--epochs",
        help="[OPT] Number of epochs to perform",
        type=int,
    )
    
    #val_interval
    parser.add_argument(
        "val_interval",
        help="Number of training steps to wait before validating the model",
        type=int
    )
    
    #mask_time_prob
    parser.add_argument(
        "mask_time_prob",
        help="Probability of masking a time step in the input sequence",
        type=float
    )
    
    #batch_size
    parser.add_argument(
        "batch_size",
        help="Batch_size",
        type=int
    )
    
    #largeness
    parser.add_argument(
        "largeness",
        help="Model largeness in {base, large, x-large}",
        type=str,
        choices=["base", "large", "small"]
    )
    
    #alpha
    parser.add_argument(
        "alpha",
        help="[OPT] Alpha weight in the pretraining loss function",
        type=float
    )
    
    #kmeans_path
    parser.add_argument(
        "kmeans_path",
        help="Path to a file that contains paths to KMeans models ",
        type=str
    )
    
    #train_features_path
    parser.add_argument(
        "train_features_path",
        help="In case of pretraining or resumed pretraing, the path from which training features to cluster can be loaded",
        type=str,
    )
    
    #val_features_path
    parser.add_argument(
        "val_features_path",
        help="In case of pretraining or resumed pretraing, the path from which validation features to cluster can be loaded",
        type=str,
    )
    
    #vocab_sizes
    parser.add_argument(
        "vocab_sizes",
        help="Vocabulary sizes, i.e. num of labels/clusters per each task/clustering model",
        type=int,
        nargs="+"
    )
    
    #patience
    parser.add_argument(
        "--patience",
        help="Patience for early stopping",
        type=int
    )
    
    #intervals_for_penalty
    parser.add_argument(
        "--intervals_for_penalty",
        help="Number of validation intervals with worsening performance to wait before penalizing model with regularization. Default 4",
        type=int,
        default=4
    )
    
    #resume_pretraining
    parser.add_argument(
        "--resume_pretraining",
        help="Whether to resume pretraing",
        action="store_true"
    )
    
    #accumulation_steps
    parser.add_argument(
        "--accumulation_steps", 
        help="[OPT] Number of batch gradients to accumulate before updating model params. Default 1",
        type=int,
        default=1
    )
    
    #downsampling_factor
    parser.add_argument(
        "--downsampling_factor",
        help="[OPT.] Integer indicating the downsampling factor of the ECG signal. Default None",
        type=int
    )

    #lr
    parser.add_argument(
        "--lr",
        help="[OPT] Learning rate. Default 5e-5",
        type=float,
        default=5e-5
    )
    
    #load_path
    parser.add_argument(
        "--load_path",
        help="[OPT] Path to a partially pretrained model in order to resume pretraining",
        type=str
    )
    
    #dynamic-reg
    parser.add_argument(
        "--dynamic_reg",
        help="[OPT] Whether to use dynamic regularization",
        action="store_true"
    )
    
    # weight_decay_mult
    parser.add_argument(
        "--weight_decay_mult",
        help="Weight decay. Default 0",
        type=int,
        default=1
    )
    
    # model_dropout_mult
    parser.add_argument(
        "--model_dropout_mult",
        help="Model dropout. Default 0",
        type=int,
        default=0
    )

    # wandb_run_name
    parser.add_argument(
        "--wandb_run_name",
        help="OPT. Wandb run name. Default none",
        type=str,
        default=None
    ) 
     

        
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available. CPU training not supported")
        exit(1)
    
    if args.train_iteration > 3 or args.train_iteration < 1:
        raise ValueError(f"Argument train_iteration must be an integer in [1, 3] range. Inserted {args.train_iteration}")
    
    if args.epochs is None and args.training_steps is None:
        raise ValueError("Argument epochs or training_steps must be provided")
    
    if args.epochs is not None and args.training_steps is not None:
        raise ValueError("Argument epochs and training_steps cannot be provided at the same time")
    
    if args.training_steps is not None and args.training_steps % args.val_interval != 0:
        raise ValueError(f"Argument training_steps must be divisible by argument val_interval. Inserted {args.training_steps} and {args.val_interval}")
    
    if args.largeness not in ["base", "large", "small"]:
        raise ValueError(f"Argument largeness must be in [base, large, x-large] range. Inserted {args.largeness}")
    
    if args.mask_time_prob < 0.0 or args.mask_time_prob > 1.0:
        raise ValueError(f"Argument mask_time_prob must be a float in [0.0, 1.0] range. Inserted {args.mask_time_prob}")
    
    if args.alpha < 0.0 or args.alpha > 1.0:
        raise ValueError(f"Argument alpha must be a float in [0.0, 1.0] and must provided if pretrain is provided. Inserted {args.alpha}")
    
    if not os.path.exists(args.kmeans_path):
        raise ValueError(f"Argument kmeans_path must be a valid path. Inserted {args.kmeans_path}")
    
    if not os.path.exists(args.train_features_path):
        raise ValueError(f"Argument train_features_path must be a valid path. Inserted {args.train_features_path}")
    
    if not os.path.exists(args.val_features_path):
        raise ValueError(f"Argument val_features_path must be a valid path. Inserted {args.val_features_path}")

    if args.resume_pretraining and args.load_path is None:
        raise ValueError("Argument load_path must be provided is argument resume_pretraining is provided")

    if args.accumulation_steps is not None and args.training_steps is not None and args.training_steps % args.accumulation_steps != 0:
        raise ValueError("Argument training_steps must be divisible by argument accumulation_steps")
    
    

    
    print("Inserted arguments: ")
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    
    
    ### NOTE: this is to test sweeps ###

    # wandb.init(entity="cardi-ai", project="ECG-pretraining", group=("self-supervised"))
    
    # args = wandb.config

    train(args)
